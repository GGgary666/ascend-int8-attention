"""Record deterministic vLLM responses and compare a relocated INT8 installation.

The server must support return_token_ids and X-data-parallel-rank. Requests use
synthetic public prompts; the saved record contains no server address or path.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import urllib.request


def cases():
    document = "\n".join(f"row {i:04d}: value {i * 7:06d};" for i in range(900))
    long_prompt = document + "\nReturn only the six-digit value for row 0899."
    return [
        ("ascii", [{"role": "user", "content": "只输出三个字符 CAT，不要输出其他内容。"}], 16),
        ("arithmetic", [{"role": "user", "content": "计算 17*23+45 的值，只输出数字。"}], 16),
        ("recall", [{"role": "user", "content": "记住数字 741852963。只回答这个数字。"}], 16),
        ("unicode", [{"role": "user", "content": "只原样输出：昇腾 INT8 注意力"}], 32),
        ("python", [{"role": "user", "content": "Write a Python function that computes the sum of squares of a list. Return only the function, with no explanation."}], 96),
        ("multiturn", [
            {"role": "user", "content": "Remember the identifier ORBIT-7341."},
            {"role": "assistant", "content": "I will remember ORBIT-7341."},
            {"role": "user", "content": "Return only the identifier."},
        ], 32),
        ("long_prefill", [{"role": "user", "content": long_prompt}], 32),
        ("prefix_reuse", [{"role": "user", "content": long_prompt}], 32),
    ]


def request(base_url, model, rank, messages, maximum, timeout):
    body = {
        "model": model, "messages": messages, "temperature": 0, "seed": 42,
        "max_tokens": maximum, "return_token_ids": True, "logprobs": True,
        "top_logprobs": 1, "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-data-parallel-rank": str(rank)},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.load(response)
    choice = payload["choices"][0]
    ids = choice.get("token_ids")
    prompt_ids = payload.get("prompt_token_ids")
    if not ids or not prompt_ids:
        raise ValueError("The server did not return prompt and generated token IDs")
    probabilities = (choice.get("logprobs") or {}).get("content") or []
    if len(probabilities) != len(ids):
        raise ValueError("Missing generated-token log probabilities")
    logprobs = [item["logprob"] for item in probabilities]
    if not all(math.isfinite(value) for value in logprobs):
        raise ValueError("Non-finite generated-token log probability")
    return {
        "prompt_token_ids": prompt_ids, "token_ids": ids,
        "text": choice["message"].get("content"),
        "finish_reason": choice["finish_reason"], "logprobs": logprobs,
        "usage": payload["usage"],
    }


def record(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for rank in range(args.dp_size):
        for name, messages, maximum in cases():
            response = request(args.base_url, args.model, rank, messages, maximum, args.timeout)
            rows.append({"case": name, "dp_rank": rank, **response})
            print(f"rank={rank} {name}: prompt={len(response['prompt_token_ids'])} "
                  f"generated={len(response['token_ids'])}", flush=True)
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "model": args.model, "temperature": 0, "seed": 42, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def compare(args):
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    expected = {(row["dp_rank"], row["case"]): row for row in baseline["rows"]}
    repeats = [json.loads(path.read_text()) for path in args.baseline_repeat]
    alternatives = [expected]
    for repeat in repeats:
        matrix = {(row["dp_rank"], row["case"]): row for row in repeat["rows"]}
        if matrix.keys() != expected.keys() or len(matrix) != len(repeat["rows"]):
            raise ValueError("Baseline repeat request matrices differ")
        if repeat["model"] != baseline["model"]:
            raise ValueError("Baseline repeat models differ")
        alternatives.append(matrix)
    actual = {(row["dp_rank"], row["case"]): row for row in candidate["rows"]}
    if not expected or len(expected) != len(baseline["rows"]) or len(actual) != len(candidate["rows"]):
        raise ValueError("Empty or duplicate response records")
    if expected.keys() != actual.keys() or baseline["model"] != candidate["model"]:
        raise ValueError("Baseline and candidate request matrices differ")
    if any(baseline[key] != candidate[key] for key in ("temperature", "seed")):
        raise ValueError("Baseline and candidate sampling configurations differ")
    rows = []
    for key, before in expected.items():
        after = actual[key]
        matched_repeat = None
        matched_key = None
        for index, matrix in enumerate(alternatives):
            reference = matrix[key]
            if all(reference[field] == after[field]
                   for field in ("prompt_token_ids", "token_ids", "text", "finish_reason")):
                before = reference
                matched_repeat = index
                matched_key = key
                break
        strict_match = matched_repeat is not None
        if not strict_match and args.allow_observed_prompt_variants:
            for index, matrix in enumerate(alternatives):
                for source_key, reference in matrix.items():
                    if all(reference[field] == after[field] for field in
                           ("prompt_token_ids", "token_ids", "text", "finish_reason")):
                        before, matched_repeat, matched_key = reference, index, source_key
                        break
                if matched_repeat is not None:
                    break
        same_ids = before["token_ids"] == after["token_ids"]
        same_prompts = before["prompt_token_ids"] == after["prompt_token_ids"]
        same_text = before["text"] == after["text"]
        same_finish = before["finish_reason"] == after["finish_reason"]
        delta = None
        if same_ids:
            if len(before["logprobs"]) != len(after["logprobs"]):
                raise ValueError("Generated-token log probability lengths differ")
            delta = max((abs(a - b) for a, b in zip(before["logprobs"], after["logprobs"])), default=0)
        rows.append({
            "case": key[1], "dp_rank": key[0], "prompt_tokens": len(before["prompt_token_ids"]),
            "generated_tokens": len(after["token_ids"]), "prompt_ids_equal": same_prompts,
            "generated_ids_equal": same_ids, "text_equal": same_text,
            "finish_reason_equal": same_finish, "maximum_logprob_absolute_diff": delta,
            "matched_baseline_record": matched_repeat,
            "matched_baseline_case": matched_key[1] if matched_key else None,
            "matched_baseline_dp_rank": matched_key[0] if matched_key else None,
            "strict_same_case_match": strict_match,
            "output_sha256": hashlib.sha256((after["text"] or "").encode()).hexdigest(),
            "candidate_cached_prompt_tokens": (after["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "passed": same_ids and same_prompts and same_text and same_finish,
        })
    report = {
        "schema_version": 1, "model": baseline["model"], "temperature": 0, "seed": 42,
        "requests": len(rows), "passed_requests": sum(row["passed"] for row in rows),
        "strict_same_case_matches": sum(row["strict_same_case_match"] for row in rows),
        "allow_observed_prompt_variants": args.allow_observed_prompt_variants,
        "observed_prompt_variant_matches": sum(
            row["passed"] and not row["strict_same_case_match"] for row in rows
        ),
        "baseline_records": len(alternatives),
        "baseline_variable_requests": sum(
            any(matrix[key]["token_ids"] != expected[key]["token_ids"] for matrix in alternatives)
            for key in expected
        ),
        "generated_tokens": sum(row["generated_tokens"] for row in rows),
        "maximum_logprob_absolute_diff": max((row["maximum_logprob_absolute_diff"] or 0 for row in rows), default=0),
        "passed": all(row["passed"] for row in rows), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    raise SystemExit(not report["passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("record")
    capture.add_argument("--base-url", required=True)
    capture.add_argument("--model", default="DeepSeek-V4-Pro")
    capture.add_argument("--dp-size", type=int, default=2)
    capture.add_argument("--timeout", type=float, default=600)
    capture.add_argument("--output", type=Path, required=True)
    diff = sub.add_parser("compare")
    diff.add_argument("--baseline", type=Path, required=True)
    diff.add_argument("--baseline-repeat", type=Path, action="append", default=[],
                      help="Additional baseline captures collected before testing the candidate")
    diff.add_argument("--allow-observed-prompt-variants", action="store_true",
                      help="Accept an exact observed baseline response for identical prompt IDs "
                           "across ranks/cache phases; report same-case mismatches separately")
    diff.add_argument("--candidate", type=Path, required=True)
    diff.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "record":
        if args.dp_size < 1:
            parser.error("--dp-size must be positive")
        record(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
