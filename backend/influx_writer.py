import os
import logging
from typing import Optional

log = logging.getLogger("thermo-twin.influx")

INFLUX_URL    = os.getenv("INFLUX_URL",    "http://localhost:8086")
INFLUX_TOKEN  = os.getenv("INFLUX_TOKEN",  "thermo-twin-dev-token")
INFLUX_ORG    = os.getenv("INFLUX_ORG",    "thermo-twin")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "hvac_data")


class InfluxWriter:
    def __init__(
        self,
        url: str = INFLUX_URL,
        token: str = INFLUX_TOKEN,
        org: str = INFLUX_ORG,
        bucket: str = INFLUX_BUCKET,
        enabled: bool = True,
    ):
        self._bucket = bucket
        self._org = org
        self._enabled = enabled
        self._client = None
        self._write_api = None

        if enabled:
            self._init_client(url, token)

    def _init_client(self, url: str, token: str) -> None:
        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import SYNCHRONOUS
            self._client = InfluxDBClient(url=url, token=token, org=self._org)
            self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
            log.info("InfluxDB connected: %s / bucket=%s", url, self._bucket)
        except Exception as exc:
            log.warning("InfluxDB unavailable (%s) — writes disabled", exc)
            self._enabled = False

    def write_sample(self, sample) -> None:
        if not self._enabled or self._write_api is None:
            return
        try:
            from influxdb_client import Point
            p = (
                Point("sensor_sample")
                .tag("machine_id", sample.machine_id)
                .field("compressor_power_kw",    sample.compressor_power_kw)
                .field("discharge_pressure_psi", sample.discharge_pressure_psi)
                .field("fan_rpm",                float(sample.fan_rpm))
                .field("supply_air_temp_c",      sample.supply_air_temp_c)
                .time(int(sample.timestamp * 1e9))
            )
            self._write_api.write(bucket=self._bucket, org=self._org, record=p)
        except Exception as exc:
            log.debug("InfluxDB write_sample failed: %s", exc)

    def write_alert(self, alert) -> None:
        if not self._enabled or self._write_api is None:
            return
        try:
            from influxdb_client import Point
            p = (
                Point("hvac_alert")
                .tag("machine_id", alert.machine_id)
                .tag("fault_type", alert.fault_type)
                .tag("action",     alert.action)
                .field("severity_score",       float(alert.severity_score))
                .field("reconstruction_error", float(alert.reconstruction_error or 0.0))
            )
            self._write_api.write(bucket=self._bucket, org=self._org, record=p)
        except Exception as exc:
            log.debug("InfluxDB write_alert failed: %s", exc)

    def close(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
