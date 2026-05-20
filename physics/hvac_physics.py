import logging
from dataclasses import dataclass

log = logging.getLogger("thermo-twin.physics")


@dataclass
class SensorPrediction:
    compressor_power_kw: float
    discharge_pressure_psi: float
    fan_rpm: float
    supply_air_temp_c: float
    model_used: str  # "coolprop" or "linear"

    def to_dict(self) -> dict:
        return {
            "compressor_power_kw":    self.compressor_power_kw,
            "discharge_pressure_psi": self.discharge_pressure_psi,
            "fan_rpm":                self.fan_rpm,
            "supply_air_temp_c":      self.supply_air_temp_c,
            "model_used":             self.model_used,
        }


@dataclass
class OperatingConditions:
    ambient_temp_c: float
    load_demand_pct: float
    compressor_speed_pct: float
    machine_id: str = "LIVE-DEMO-UNIT"


class HVACPhysicsModel:
    """
    First-principles HVAC thermodynamic simulator.

    Uses CoolProp R-410A refrigerant properties when available; falls back to
    calibrated linear equations matching the synthetic training data exactly.

    Commissioning baselines (CARRIER-CHILLER-01):
        k_disc   = 69.99   (discharge_pressure / compressor_power)
        k_fan    = 340.19  (fan_rpm / compressor_power)
        k_temp_a = 18.05   (supply_air_temp intercept)
        k_temp_b = -2.01   (supply_air_temp slope vs compressor_power)
    """

    _NOMINAL_POWER_KW   = 3.5
    _NOMINAL_AMBIENT_C  = 35.0
    _NOMINAL_SPEED_PCT  = 70.0

    # Isentropic efficiency curve: η = a0 + a1*s + a2*s² — gives η≈0.745 at s=0.70
    _ETA_A0 = 0.30
    _ETA_A1 = 1.45
    _ETA_A2 = -0.70

    _COND_TEMP_SENSITIVITY = 0.012  # psi per °C above nominal ambient

    # Commissioning coefficients
    _K_DISC   = 69.99
    _K_FAN    = 340.19
    _K_TEMP_A = 18.05
    _K_TEMP_B = -2.01

    def __init__(self, use_coolprop: bool = True):
        self._coolprop_available = False
        if use_coolprop:
            self._coolprop_available = self._check_coolprop()
        # Promote class-level commissioning coefficients to instance attributes
        # so Phase 7's recalibration can mutate them without touching the class.
        self._K_DISC   = type(self)._K_DISC
        self._K_FAN    = type(self)._K_FAN
        self._K_TEMP_A = type(self)._K_TEMP_A
        self._K_TEMP_B = type(self)._K_TEMP_B

    def set_calibration(self, **kwargs) -> None:
        """Update one or more commissioning coefficients at runtime.

        Accepts any subset of: k_disc, k_fan, k_temp_a, k_temp_b. Values must
        be positive. Used by the Phase 7 recalibration scheduler after a
        Bayesian re-fit so future predictions reflect the updated calibration.
        """
        mapping = {
            "k_disc":   "_K_DISC",
            "k_fan":    "_K_FAN",
            "k_temp_a": "_K_TEMP_A",
            "k_temp_b": "_K_TEMP_B",
        }
        for key, val in kwargs.items():
            attr = mapping.get(key)
            if attr is None or val is None:
                continue
            v = float(val)
            if v <= 0:
                continue
            setattr(self, attr, v)

    def get_calibration(self) -> dict:
        return {
            "k_disc":   self._K_DISC,
            "k_fan":    self._K_FAN,
            "k_temp_a": self._K_TEMP_A,
            "k_temp_b": self._K_TEMP_B,
        }

    def _check_coolprop(self) -> bool:
        try:
            import CoolProp.CoolProp as CP
            CP.PropsSI("T", "P", 101325, "Q", 0, "R410A")  # probe call to verify runtime works
            log.info("CoolProp available — using R-410A refrigerant properties")
            return True
        except Exception as exc:
            log.info("CoolProp unavailable (%s) — using linear physics fallback", type(exc).__name__)
            return False

    def predict(self, conditions: OperatingConditions) -> SensorPrediction:
        if self._coolprop_available:
            return self._predict_coolprop(conditions)
        return self._predict_linear(conditions)

    def _predict_coolprop(self, c: OperatingConditions) -> SensorPrediction:
        try:
            import CoolProp.CoolProp as CP
            FLUID = "R410A"

            speed_frac = c.compressor_speed_pct / 100.0
            load_frac  = c.load_demand_pct / 100.0
            eta        = self._isentropic_efficiency(speed_frac)

            # Evaporator suction (saturated vapour)
            T_evap_k = 273.15 + 5.0 - (1.0 - load_frac) * 5.0
            P_evap   = CP.PropsSI("P", "T", T_evap_k, "Q", 1.0, FLUID)
            h1       = CP.PropsSI("H", "T", T_evap_k, "Q", 1.0, FLUID)
            s1       = CP.PropsSI("S", "T", T_evap_k, "Q", 1.0, FLUID)

            # Condenser discharge
            T_cond_approach = 15.0 + (c.ambient_temp_c - self._NOMINAL_AMBIENT_C) * 0.3
            T_cond_k        = 273.15 + c.ambient_temp_c + T_cond_approach
            P_cond          = CP.PropsSI("P", "T", T_cond_k, "Q", 1.0, FLUID)

            h2s = CP.PropsSI("H", "P", P_cond, "S", s1, FLUID)
            h2  = h1 + (h2s - h1) / eta

            m_dot    = 0.08 * speed_frac * (0.5 + 0.5 * load_frac)
            power_kw = m_dot * (h2 - h1) / 1000.0

            pressure_psi = P_cond / 6894.76

            fan_rpm = self._K_FAN * (power_kw / self._NOMINAL_POWER_KW) * (
                1.0 + 0.02 * (c.ambient_temp_c - self._NOMINAL_AMBIENT_C)
            )
            supply_temp_c = self._K_TEMP_A + self._K_TEMP_B * power_kw + (
                (c.ambient_temp_c - self._NOMINAL_AMBIENT_C) * 0.08
            )

            return SensorPrediction(
                compressor_power_kw    = round(max(0.5, power_kw), 4),
                discharge_pressure_psi = round(max(50.0, pressure_psi), 2),
                fan_rpm                = round(max(100.0, fan_rpm), 1),
                supply_air_temp_c      = round(supply_temp_c, 2),
                model_used             = "coolprop",
            )

        except Exception as exc:
            log.warning("CoolProp prediction failed (%s) — using linear fallback", exc)
            return self._predict_linear(c)

    def _isentropic_efficiency(self, speed_frac: float) -> float:
        eta = self._ETA_A0 + self._ETA_A1 * speed_frac + self._ETA_A2 * speed_frac ** 2
        return max(0.30, min(0.85, eta))

    def _predict_linear(self, c: OperatingConditions) -> SensorPrediction:
        # speed_ratio = 1.0 at nominal speed (70%), so power = nominal at nominal conditions
        speed_ratio    = c.compressor_speed_pct / self._NOMINAL_SPEED_PCT
        load_frac      = c.load_demand_pct / 100.0
        demand_offset  = (load_frac - 0.5) * 0.8
        power_kw       = max(2.0, min(6.0, self._NOMINAL_POWER_KW * speed_ratio + demand_offset))

        ambient_delta  = c.ambient_temp_c - self._NOMINAL_AMBIENT_C
        pressure_psi   = self._K_DISC  * power_kw + ambient_delta * self._COND_TEMP_SENSITIVITY * self._K_DISC
        fan_rpm        = self._K_FAN   * power_kw
        supply_temp_c  = self._K_TEMP_A + self._K_TEMP_B * power_kw - ambient_delta * 0.05

        return SensorPrediction(
            compressor_power_kw    = round(power_kw, 4),
            discharge_pressure_psi = round(pressure_psi, 2),
            fan_rpm                = round(fan_rpm, 1),
            supply_air_temp_c      = round(supply_temp_c, 2),
            model_used             = "linear",
        )
