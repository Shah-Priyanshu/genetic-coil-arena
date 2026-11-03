// R18 desk bench for the OpenSwarm brains. One firmware on every board; commands arrive
// over USB serial (115200) and results go back as "#TAG,..." CSV lines that
// scripts/r18_bench.py collects.
//   info                         MAC, known peers, role
//   hello                        broadcast a hello so the other boards learn this one
//   ping <count> <bytes> <ms>    broadcast pings (as OpenSwarm sends every packet); every
//                                board echoes by broadcast; prints #RTT and #GOT lines
//   ap                           become the Wi-Fi FTM responder (soft AP, channel 1)
//   ftm <count>                  join the responder and range to it; prints #FTM lines
//   brain <calls>                golden check of both int8 brains, then time them
//   gas <count> <ms>             read the SGP40 (VOC index and raw); prints #GAS lines
#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>
#include <esp_now.h>
#include <esp_timer.h>
#include <esp_wifi.h>

#include "Adafruit_SGP40.h"
#include "golden.h"
#include "swarm_brains_int8.h"

static const char *FTM_SSID = "gca-bench";
static const char *FTM_PASS = "gca-bench-ftm";
static const int CHANNEL = 1;
static const uint8_t BCAST[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
static const uint8_t MAGIC = 0xA7;
static const int MAX_PEERS = 8, MAX_SEQ = 2000;

struct __attribute__((packed)) Head {
  uint8_t magic, kind;  // 'H' hello, 'P' ping, 'R' reply
  uint16_t run;
  uint32_t seq;
  int64_t t_us;  // sender's clock when the ping left
  uint32_t got;  // in a reply: pings of this run the replier has received
};
struct Reply {
  Head h;
  uint8_t len;
};

static uint8_t peers[MAX_PEERS][6];
static int n_peers = 0;
static uint16_t got_run[MAX_PEERS];
static uint32_t got_count[MAX_PEERS];
static int32_t rtt_us[MAX_PEERS][MAX_SEQ];  // replies to this board's pings, -1 = none
static uint32_t their_got[MAX_PEERS];
static uint16_t my_run = 0;
static QueueHandle_t replies;
static SemaphoreHandle_t ftm_done;
static volatile bool ftm_ok = false;
static volatile uint32_t ftm_cm = 0, ftm_ns = 0;
static volatile uint8_t ftm_status = 0;
static bool is_ap = false;
static Adafruit_SGP40 sgp;
static bool sgp_ready = false;

static int peer_index(const uint8_t *mac) {
  for (int i = 0; i < n_peers; i++)
    if (!memcmp(peers[i], mac, 6)) return i;
  if (n_peers == MAX_PEERS) return -1;
  memcpy(peers[n_peers], mac, 6);
  got_run[n_peers] = 0xffff;
  return n_peers++;
}

static void mac_str(const uint8_t *mac, char *out) {
  sprintf(out, "%02X:%02X:%02X:%02X:%02X:%02X", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

static void on_recv(const uint8_t *mac, const uint8_t *data, int len) {
  if (len < (int)sizeof(Head)) return;
  Head h;
  memcpy(&h, data, sizeof(Head));
  if (h.magic != MAGIC) return;
  int p = peer_index(mac);
  if (p < 0) return;
  if (h.kind == 'P') {
    if (got_run[p] != h.run) {
      got_run[p] = h.run;
      got_count[p] = 0;
    }
    Reply r;
    r.h = h;
    r.h.kind = 'R';
    r.h.got = ++got_count[p];
    r.len = (uint8_t)len;
    xQueueSend(replies, &r, 0);  // runs in the Wi-Fi task, not an interrupt
  } else if (h.kind == 'R' && h.run == my_run && h.seq < MAX_SEQ) {
    rtt_us[p][h.seq] = (int32_t)(esp_timer_get_time() - h.t_us);
    if (h.got > their_got[p]) their_got[p] = h.got;
  }
}

static void send_bytes(const Head &h, int len) {
  static uint8_t buf[250];
  memset(buf, 0, sizeof(buf));
  memcpy(buf, &h, sizeof(Head));
  esp_now_send(BCAST, buf, len);
}

static void on_ftm(arduino_event_t *event) {
  wifi_event_ftm_report_t *r = &event->event_info.wifi_ftm_report;
  ftm_status = r->status;
  ftm_ok = r->status == FTM_STATUS_SUCCESS;
  ftm_cm = r->dist_est;
  ftm_ns = r->rtt_est;
  if (r->ftm_report_data) free(r->ftm_report_data);
  xSemaphoreGive(ftm_done);
}

static void start_now() {
  esp_now_deinit();
  esp_now_init();
  esp_now_register_recv_cb(on_recv);
  esp_now_peer_info_t info = {};
  memcpy(info.peer_addr, BCAST, 6);
  info.channel = 0;  // the radio's current channel
  info.ifidx = WIFI_IF_STA;
  esp_now_add_peer(&info);
}

// --------------------------------------------------------------------- brains
static int32_t brain(const int8_t w0[BRAIN_IN][BRAIN_HIDDEN], const int32_t *b0, float s0,
                     const int8_t w1[BRAIN_HIDDEN][1], const int32_t *b1, const int8_t *x) {
  int8_t h[BRAIN_HIDDEN];
  for (int j = 0; j < BRAIN_HIDDEN; j++) {
    int32_t acc = b0[j];
    for (int i = 0; i < BRAIN_IN; i++) acc += (int32_t)x[i] * w0[i][j];
    float q = roundf(tanhf(acc * s0) * 127.0f);
    h[j] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
  }
  int32_t out = b1[0];
  for (int j = 0; j < BRAIN_HIDDEN; j++) out += (int32_t)h[j] * w1[j][0];
  return out;
}

static int32_t gas(const int8_t *x) { return brain(gas_w0, gas_b0, gas_scale0, gas_w1, gas_b1, x); }
static int32_t cam(const int8_t *x) { return brain(cam_w0, cam_b0, cam_scale0, cam_w1, cam_b1, x); }

static void cmd_brain(int calls) {
  int ok_gas = 0, ok_cam = 0, worst = 0;
  for (int k = 0; k < GOLDEN_N; k++) {
    int32_t g = gas(golden_x[k]), c = cam(golden_x[k]);
    ok_gas += g == golden_gas[k];
    ok_cam += c == golden_cam[k];
    worst = max(worst, (int)max(abs(g - golden_gas[k]), abs(c - golden_cam[k])));
  }
  Serial.printf("#GOLDEN,%d,%d,%d,%d\n", GOLDEN_N, ok_gas, ok_cam, worst);
  static int8_t xs[64][BRAIN_IN];
  for (int k = 0; k < 64; k++)
    for (int i = 0; i < BRAIN_IN; i++) xs[k][i] = (int8_t)(esp_random() % 255 - 127);
  volatile int32_t sink = 0;
  for (int which = 0; which < 2; which++) {
    int64_t t0 = esp_timer_get_time();
    for (int n = 0; n < calls; n++) sink += which ? cam(xs[n & 63]) : gas(xs[n & 63]);
    int64_t dt = esp_timer_get_time() - t0;
    Serial.printf("#BRAIN,%s,%d,%.3f\n", which ? "cam" : "gas", calls, (double)dt / calls);
  }
}

// ----------------------------------------------------------------- commands
static void cmd_info() {
  uint8_t mac[6];
  char s[18];
  esp_wifi_get_mac(WIFI_IF_STA, mac);
  mac_str(mac, s);
  Serial.printf("#INFO,%s,%s,%d\n", s, is_ap ? "ap" : "sta", n_peers);
  for (int i = 0; i < n_peers; i++) {
    mac_str(peers[i], s);
    Serial.printf("#PEER,%d,%s\n", i, s);
  }
}

static void cmd_ping(int count, int bytes, int interval_ms) {
  count = constrain(count, 1, MAX_SEQ);
  bytes = constrain(bytes, (int)sizeof(Head), 250);
  my_run++;
  for (int p = 0; p < MAX_PEERS; p++) {
    their_got[p] = 0;
    for (int s = 0; s < MAX_SEQ; s++) rtt_us[p][s] = -1;
  }
  for (int s = 0; s < count; s++) {
    Head h = {MAGIC, 'P', my_run, (uint32_t)s, esp_timer_get_time(), 0};
    send_bytes(h, bytes);
    int64_t until = esp_timer_get_time() + interval_ms * 1000LL;
    while (esp_timer_get_time() < until) {
      Reply r;
      if (xQueueReceive(replies, &r, 0) == pdTRUE) send_bytes(r.h, r.len);
    }
  }
  delay(300);
  char s[18];
  for (int p = 0; p < n_peers; p++) {
    mac_str(peers[p], s);
    for (int q = 0; q < count; q++)
      if (rtt_us[p][q] >= 0) Serial.printf("#RTT,%u,%s,%d,%d\n", my_run, s, q, rtt_us[p][q]);
    Serial.printf("#GOT,%u,%s,%u,%d,%d,%d\n", my_run, s, their_got[p], count, bytes, interval_ms);
  }
}

static void cmd_ap() {
  WiFi.mode(WIFI_AP_STA);
  WiFi.softAP(FTM_SSID, FTM_PASS, CHANNEL, 0, 4, true);
  is_ap = true;
  start_now();
  Serial.println("#AP,ok");
}

static void cmd_ftm(int count) {
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.begin(FTM_SSID, FTM_PASS);
    for (int i = 0; i < 100 && WiFi.status() != WL_CONNECTED; i++) delay(100);
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("#FTM_ERR,no responder (send 'ap' to one board first)");
      return;
    }
    start_now();
  }
  for (int i = 0; i < count; i++) {
    if (!WiFi.initiateFTM(16, 2)) {
      Serial.printf("#FTM,%d,start_failed,0,0\n", i);
      continue;
    }
    if (xSemaphoreTake(ftm_done, pdMS_TO_TICKS(3000)) != pdTRUE) {
      Serial.printf("#FTM,%d,timeout,0,0\n", i);
      continue;
    }
    Serial.printf("#FTM,%d,%s,%u,%u\n", i, ftm_ok ? "ok" : "fail", ftm_cm, ftm_ns);
  }
}

static void cmd_gas(int count, int period_ms) {
  if (!sgp_ready) {
    Wire.begin();
    sgp_ready = sgp.begin(&Wire);
    if (!sgp_ready) {
      Serial.println("#GAS_ERR,no SGP40 on I2C (SDA 8, SCL 9)");
      return;
    }
  }
  for (int i = 0; i < count; i++) {
    int64_t t = esp_timer_get_time();
    uint16_t raw = sgp.measureRaw(25, 50);
    int32_t voc = sgp.measureVocIndex(25, 50);
    Serial.printf("#GAS,%d,%lld,%u,%d\n", i, (long long)(t / 1000), raw, voc);
    delay(period_ms);
  }
}

static void handle(char *line) {
  char *cmd = strtok(line, " ");
  if (!cmd) return;
  auto arg = [](int fallback) {
    char *t = strtok(NULL, " ");
    return t ? atoi(t) : fallback;
  };
  if (!strcmp(cmd, "info")) cmd_info();
  else if (!strcmp(cmd, "hello")) {
    Head h = {MAGIC, 'H', 0, 0, esp_timer_get_time(), 0};
    send_bytes(h, sizeof(Head));
    Serial.println("#HELLO,sent");
  } else if (!strcmp(cmd, "ping")) {
    int count = arg(100), bytes = arg(32), ms = arg(20);
    cmd_ping(count, bytes, ms);
  } else if (!strcmp(cmd, "ap")) cmd_ap();
  else if (!strcmp(cmd, "ftm")) cmd_ftm(arg(20));
  else if (!strcmp(cmd, "brain")) cmd_brain(arg(10000));
  else if (!strcmp(cmd, "gas")) {
    int count = arg(60), ms = arg(1000);
    cmd_gas(count, ms);
  } else Serial.printf("#ERR,unknown command %s\n", cmd);
  Serial.println("#DONE");
}

void setup() {
  Serial.begin(115200);
  replies = xQueueCreate(64, sizeof(Reply));
  ftm_done = xSemaphoreCreateBinary();
  WiFi.onEvent(on_ftm, ARDUINO_EVENT_WIFI_FTM_REPORT);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  esp_wifi_set_channel(CHANNEL, WIFI_SECOND_CHAN_NONE);
  start_now();
  Serial.println("#BOOT,r18-bench");
}

void loop() {
  static char line[96];
  static int n = 0;
  Reply r;
  while (xQueueReceive(replies, &r, 0) == pdTRUE) send_bytes(r.h, r.len);
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (n) {
        line[n] = 0;
        handle(line);
        n = 0;
      }
    } else if (n < (int)sizeof(line) - 1) {
      line[n++] = c;
    }
  }
}
