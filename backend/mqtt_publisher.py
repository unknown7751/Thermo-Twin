import os
import json
import logging

log = logging.getLogger("thermo-twin.mqtt")

MQTT_HOST      = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT      = int(os.getenv("MQTT_PORT", "1883"))
MQTT_KEEPALIVE = 60
TOPIC_SAMPLES  = "thermo/samples/{machine_id}"
TOPIC_ALERTS   = "thermo/alerts/{machine_id}"


class MQTTPublisher:
    def __init__(
        self,
        host: str = MQTT_HOST,
        port: int = MQTT_PORT,
        enabled: bool = True,
    ):
        self._host    = host
        self._port    = port
        self._enabled = enabled
        self._client  = None

        if enabled:
            self._init_client()

    def _init_client(self) -> None:
        try:
            import paho.mqtt.client as mqtt
            self._client = mqtt.Client(client_id="thermo-twin-backend", clean_session=True)
            self._client.on_connect    = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.connect_async(self._host, self._port, MQTT_KEEPALIVE)
            self._client.loop_start()
            log.info("MQTT connecting to %s:%d", self._host, self._port)
        except Exception as exc:
            log.warning("MQTT unavailable (%s) — publishing disabled", exc)
            self._enabled = False

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            log.info("MQTT connected")
        else:
            log.warning("MQTT connect failed rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        if rc != 0:
            log.warning("MQTT unexpected disconnect rc=%d", rc)

    def publish_sample(self, sample) -> None:
        if not self._enabled or self._client is None:
            return
        topic = TOPIC_SAMPLES.format(machine_id=sample.machine_id)
        try:
            self._client.publish(topic, json.dumps(sample.to_dict()), qos=0)
        except Exception as exc:
            log.debug("MQTT publish_sample failed: %s", exc)

    def publish_alert(self, alert) -> None:
        if not self._enabled or self._client is None:
            return
        topic = TOPIC_ALERTS.format(machine_id=alert.machine_id)
        try:
            self._client.publish(topic, json.dumps(alert.to_dict()), qos=1)
        except Exception as exc:
            log.debug("MQTT publish_alert failed: %s", exc)

    def disconnect(self) -> None:
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
