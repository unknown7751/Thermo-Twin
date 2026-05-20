from dataclasses import dataclass
from physics.hvac_physics import SensorPrediction


@dataclass
class ComponentHealth:
    """
    Current health state of all three modelled components.
    All values 0–100, where 100 = new/perfect condition.
    """
    refrigerant_charge_pct: float = 100.0
    compressor_efficiency_pct: float = 100.0
    fan_health_pct: float = 100.0

    def to_dict(self) -> dict:
        return {
            "refrigerant_charge_pct":    self.refrigerant_charge_pct,
            "compressor_efficiency_pct": self.compressor_efficiency_pct,
            "fan_health_pct":            self.fan_health_pct,
        }


class DegradationModel:
    """
    Applies component health degradation offsets to a healthy SensorPrediction.

    Magnitudes calibrated from fault_injector.py:
        Refrigerant: 38% pressure drop at 0% charge, +7°C temp at 0% charge
        Fan bearing: 80% RPM loss at 0% health + cascading compressor load
        Compressor:  +2.0 kW power, -45 psi pressure, +3.0°C temp at 0% efficiency
    """

    _REFRIG_PRESSURE_LOSS_FRAC = 0.38   # fraction of nominal pressure lost at 0% charge
    _REFRIG_TEMP_RISE_C        = 7.0    # °C rise at 0% charge
    _FAN_RPM_LOSS_FRAC         = 0.80   # fraction of nominal RPM lost at 0% fan health
    _FAN_POWER_CASCADE_KW      = 2.0    # extra kW at 0% fan health (cascade)
    # Condenser fan failure traps rejected heat → discharge (head) pressure
    # SPIKES. Modeled as a fraction of nominal pressure that grows with the
    # SQUARE of fan loss (heat-rejection capacity falls non-linearly as RPM
    # drops). At 0% fan health this raises head pressure ~+75%.
    _FAN_PRESSURE_RISE_FRAC    = 0.75
    _FAN_TEMP_CASCADE_C        = 4.5    # °C rise at 0% fan health
    _COMP_POWER_GAIN_KW        = 2.0    # extra kW at 0% compressor efficiency
    _COMP_PRESSURE_LOSS_PSI    = 45.0   # psi drop at 0% compressor efficiency
    _COMP_TEMP_RISE_C          = 3.0    # °C rise at 0% compressor efficiency

    def apply(self, healthy: SensorPrediction, health: ComponentHealth) -> SensorPrediction:
        """Apply degradation offsets to a healthy sensor prediction."""
        power    = healthy.compressor_power_kw
        pressure = healthy.discharge_pressure_psi
        fan_rpm  = healthy.fan_rpm
        temp     = healthy.supply_air_temp_c

        # Refrigerant charge depletion
        charge_loss = (100.0 - health.refrigerant_charge_pct) / 100.0
        pressure   -= self._REFRIG_PRESSURE_LOSS_FRAC * charge_loss * pressure
        temp       += self._REFRIG_TEMP_RISE_C * charge_loss

        # Fan bearing wear: RPM collapses, and because the condenser can no
        # longer reject heat, discharge pressure SPIKES (dominant effect),
        # compressor works harder (power up), and supply temp rises.
        fan_loss  = (100.0 - health.fan_health_pct) / 100.0
        fan_rpm  -= self._FAN_RPM_LOSS_FRAC * fan_loss * healthy.fan_rpm
        power    += self._FAN_POWER_CASCADE_KW * fan_loss * 0.7
        pressure += self._FAN_PRESSURE_RISE_FRAC * (fan_loss ** 2) * pressure
        temp     += self._FAN_TEMP_CASCADE_C * fan_loss * 0.7

        # Compressor efficiency loss
        comp_loss = (100.0 - health.compressor_efficiency_pct) / 100.0
        power    += self._COMP_POWER_GAIN_KW       * comp_loss
        pressure -= self._COMP_PRESSURE_LOSS_PSI   * comp_loss
        temp     += self._COMP_TEMP_RISE_C         * comp_loss

        return SensorPrediction(
            compressor_power_kw    = round(max(0.5, power), 4),
            discharge_pressure_psi = round(max(50.0, pressure), 2),
            fan_rpm                = round(max(0.0, fan_rpm), 1),
            supply_air_temp_c      = round(temp, 2),
            model_used             = healthy.model_used,
        )

    def advance_time(
        self,
        health: ComponentHealth,
        dt_hours: float,
        leak_rate: float = 0.0,
        fan_wear_rate: float = 0.0,
        comp_wear_rate: float = 0.0,
    ) -> ComponentHealth:
        """Advance component health by dt_hours under given wear rates (pct/hour). Returns new instance."""
        return ComponentHealth(
            refrigerant_charge_pct    = max(0.0, health.refrigerant_charge_pct    - leak_rate      * dt_hours),
            compressor_efficiency_pct = max(0.0, health.compressor_efficiency_pct - comp_wear_rate * dt_hours),
            fan_health_pct            = max(0.0, health.fan_health_pct            - fan_wear_rate  * dt_hours),
        )
