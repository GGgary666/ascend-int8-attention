"""Run the retained capacity benchmark with BF16 and INT8 graph roles.

Uses the original activation fixtures, precision gates, LSE policy and paired
measurement protocol. This module requires the source checkout and a free NPU.
"""


if __name__ == "__main__":
    from tests.npu.test_capacity import main

    main()
