import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPDMuxValidation(unittest.TestCase):
    @staticmethod
    def _args(**kwargs):
        return ServerArgs(
            model_path="dummy",
            served_model_name="dummy",
            enable_pdmux=True,
            disable_overlap_schedule=True,
            chunked_prefill_size=-1,
            **kwargs,
        )

    def test_pdmux_accepts_non_speculative_configuration(self):
        self._args().check_server_args()

    def test_pdmux_rejects_speculative_algorithms(self):
        for algorithm in ("EAGLE", "EAGLE3", "NGRAM"):
            with self.subTest(algorithm=algorithm):
                with self.assertRaisesRegex(ValueError, "does not support speculative decoding"):
                    self._args(speculative_algorithm=algorithm).check_server_args()


if __name__ == "__main__":
    unittest.main()
