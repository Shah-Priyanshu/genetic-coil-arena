"""Uplink byte budgets and the LoRaWAN EU868 numbers behind them.

A node message is ``budget`` bytes, each byte one symbol of a V = 256 vocabulary, so the
channel is ``MessageSpec.discrete(length=budget, vocab_size=256)`` (a ``"none"`` channel
at 0 bytes). The reference numbers are LoRaWAN EU868 with 125 kHz channels and a 1% duty
cycle: the largest application payload is 51 B at SF12 (DR0) and 222 B at SF7 (DR5), and
each uplink adds 13 B of MAC overhead (MHDR 1, FHDR 7, FPort 1, MIC 4).
"""

import math

from ...messages import MessageSpec

VOCAB_SIZE = 256
BUDGETS = (0, 8, 16, 51)  # Bytes per node message studied in step 1
MAX_BUDGET = 222  # EU868 SF7 maximum application payload
MAC_OVERHEAD_BYTES = 13
EU868_MAX_PAYLOAD = {7: 222, 8: 222, 9: 115, 10: 51, 11: 51, 12: 51}
DUTY_CYCLE = 0.01


def message_spec(budget: int, n_nodes: int) -> MessageSpec:
    """The uplink channel for a per-node byte budget.

    Raises:
        ValueError: If the budget is outside [0, MAX_BUDGET]
    """
    if not 0 <= int(budget) <= MAX_BUDGET:
        raise ValueError(f"budget must lie in [0, {MAX_BUDGET}] bytes, got {budget}")
    if budget == 0:
        return MessageSpec("none", n_receivers=n_nodes)
    return MessageSpec.discrete(int(budget), VOCAB_SIZE, n_receivers=n_nodes)


def lorawan_airtime_s(
    payload_bytes: int,
    spreading_factor: int = 12,
    bandwidth_hz: float = 125_000.0,
    coding_rate: int = 1,
    preamble_symbols: int = 8,
    mac_overhead: int = MAC_OVERHEAD_BYTES,
) -> float:
    """Time on air of one uplink in seconds (Semtech SX127x formula).

    Args:
        payload_bytes: Application payload
        spreading_factor: 7..12
        bandwidth_hz: Channel bandwidth
        coding_rate: 1..4 for CR 4/5..4/8
        preamble_symbols: Preamble length
        mac_overhead: Bytes added by the LoRaWAN MAC
    """
    sf = int(spreading_factor)
    t_sym = (2**sf) / bandwidth_hz
    low_data_rate = 1 if t_sym > 0.016 else 0
    pl = payload_bytes + mac_overhead
    numerator = 8 * pl - 4 * sf + 28 + 16  # explicit header, CRC on
    n_payload = 8 + max(
        math.ceil(numerator / (4 * (sf - 2 * low_data_rate))) * (coding_rate + 4), 0
    )
    return (preamble_symbols + 4.25) * t_sym + n_payload * t_sym


def min_uplink_period_s(
    payload_bytes: int, spreading_factor: int = 12, duty_cycle: float = DUTY_CYCLE
) -> float:
    """Shortest legal time between uplinks of one device under the duty cycle."""
    return lorawan_airtime_s(payload_bytes, spreading_factor) / duty_cycle


def fits_lorawan(payload_bytes: int, spreading_factor: int) -> bool:
    """Whether the payload fits one EU868 uplink at that spreading factor."""
    return payload_bytes <= EU868_MAX_PAYLOAD[spreading_factor]
