import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDecodePreallocProgress(unittest.TestCase):
    def test_prealloc_budget_is_capped_by_concrete_capacity(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.scheduler = SimpleNamespace(
            enable_hisparse=False,
            tp_worker=SimpleNamespace(is_hybrid_swa=False),
        )
        queue.tree_cache = SimpleNamespace(evictable_size=lambda: 3)
        allocator = MagicMock()
        allocator.available_size.return_value = 4
        allocator.prealloc_fits.return_value = True
        queue.token_to_kv_pool_allocator = allocator

        with patch(
            "sglang.srt.disaggregation.decode.get_disagg",
            return_value=SimpleNamespace(disaggregation_decode_enable_radix_cache=True),
        ):
            self.assertTrue(
                queue._prealloc_reservation_fits(
                    5,
                    0,
                    full_allocatable_tokens=100,
                    swa_allocatable_tokens=None,
                )
            )

        allocator.prealloc_fits.assert_called_once_with(
            queue.tree_cache,
            5,
            0,
            full_budget_tokens=7,
            swa_budget_tokens=None,
        )

    def test_reserved_transfers_progress_while_retracted_request_waits(self):
        reserved = object()
        transferred = object()
        prealloc_queue = MagicMock(retracted_queue=[object()])
        prealloc_queue.resume_retracted_reqs.return_value = []
        prealloc_queue.pop_preallocated.return_value = ([reserved], [])
        transfer_queue = MagicMock()
        transfer_queue.pop_transferred.return_value = [transferred]
        scheduler = SimpleNamespace(
            scheduler_stage_metrics=None,
            enable_decode_hicache=False,
            enable_hisparse=False,
            disagg_decode_prealloc_queue=prealloc_queue,
            disagg_decode_transfer_queue=transfer_queue,
            waiting_queue=[],
        )
        config = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_polling_interval=1,
            disaggregation_decode_host_receive_threshold=1,
        )

        with patch("sglang.srt.disaggregation.decode.get_disagg", return_value=config):
            SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)

        transfer_queue.resolve_deferred_releases.assert_called_once_with()
        prealloc_queue.pop_preallocated.assert_called_once_with()
        transfer_queue.extend.assert_called_once_with([reserved])
        transfer_queue.pop_transferred.assert_called_once_with()
        self.assertEqual(scheduler.waiting_queue, [transferred])


if __name__ == "__main__":
    unittest.main()
