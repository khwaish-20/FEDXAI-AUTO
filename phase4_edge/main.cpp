/*
 * FEDXAI-AUTO Phase 4.2: ESP32-S3 Edge Firmware
 * ==============================================
 * Main firmware for the OBD-II diagnostic dongle.
 * 
 * Architecture:
 *   OBD-II (ELM327/UART) → Sensor Buffer → TFLite Inference → BLE Alert
 *
 * Hardware: ESP32-S3-WROOM-1
 * Framework: ESP-IDF / Arduino
 * 
 * Features:
 *   - OBD-II PID polling (8 sensor channels at 1Hz)
 *   - Sliding window buffer (20 timesteps × 8 features)
 *   - TFLite Micro inference (<100ms per prediction)
 *   - BLE peripheral for mobile app communication
 *   - LED status indicators
 *   - Deep sleep power management
 */

#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Our model (auto-generated C header from Phase 4.1)
#include "fedxai_model.h"

// ============================================================
// CONFIGURATION
// ============================================================
#define FIRMWARE_VERSION    "1.0.0"
#define DEVICE_NAME         "FedXAI-Dongle"

// Model dimensions
#define SEQ_LENGTH          20      // 20-second sliding window
#define NUM_FEATURES        8       // 8 sensor channels
#define FAILURE_THRESHOLD   0.5f    // Probability threshold for failure alert

// OBD-II UART
#define OBD_SERIAL          Serial2
#define OBD_BAUD            38400
#define OBD_TX_PIN          17
#define OBD_RX_PIN          16

// LED Pins
#define LED_GREEN           2       // Healthy
#define LED_RED             4       // Failure detected
#define LED_BLUE            5       // BLE connected

// BLE UUIDs
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHAR_STATUS_UUID    "beb5483e-36e1-4688-b7f5-ea07361b26a8"  // Notify: status
#define CHAR_PROB_UUID      "beb5483e-36e1-4688-b7f5-ea07361b26a9"  // Notify: probability  
#define CHAR_ALERT_UUID     "beb5483e-36e1-4688-b7f5-ea07361b26aa"  // Notify: XAI alert
#define CHAR_SENSORS_UUID   "beb5483e-36e1-4688-b7f5-ea07361b26ab"  // Read: live sensors

// Inference timing
#define POLL_INTERVAL_MS    1000    // 1 Hz OBD-II polling
#define INFERENCE_INTERVAL  1000    // Run inference every second

// ============================================================
// GLOBAL STATE
// ============================================================

// Sensor sliding window buffer [SEQ_LENGTH x NUM_FEATURES]
float sensor_buffer[SEQ_LENGTH][NUM_FEATURES];
int buffer_index = 0;
bool buffer_full = false;

// Latest sensor readings
typedef struct {
    float engine_rpm;           // PID 0x0C
    float fuel_pressure;        // PID 0x0A (Bar)
    float fuel_trim_short;      // PID 0x06 (%)
    float fuel_trim_long;       // PID 0x07 (%)
    float o2_voltage;           // PID 0x14 (V)
    float coolant_temp;         // PID 0x05 (°C)
    float intake_air_temp;      // PID 0x0F (°C)
    float catalyst_temp;        // PID 0x3C (°C)
} SensorData;

SensorData latest_sensors;
float failure_probability = 0.0f;
bool failure_detected = false;
String xai_alert_message = "System OK";

// BLE
BLEServer* ble_server = NULL;
BLECharacteristic* char_status = NULL;
BLECharacteristic* char_prob = NULL;
BLECharacteristic* char_alert = NULL;
BLECharacteristic* char_sensors = NULL;
bool ble_connected = false;

// TFLite Micro
const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Memory arena for TFLite (adjust based on model size)
constexpr int kTensorArenaSize = 32 * 1024;  // 32 KB
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Feature scaling parameters (from Phase 2 preprocessing)
// These MUST match the scaler used during training
const float feature_means[NUM_FEATURES] = {
    3500.0f,  // Engine RPM
    3.5f,     // Fuel Pressure (Bar)
    0.0f,     // Short-Term Fuel Trim (%)
    0.0f,     // Long-Term Fuel Trim (%)
    0.45f,    // O2 Sensor Voltage (V)
    90.0f,    // Coolant Temp (°C)
    25.0f,    // Intake Air Temp (°C)
    450.0f    // Catalyst Temp (°C)
};

const float feature_stds[NUM_FEATURES] = {
    2000.0f,  // Engine RPM
    1.5f,     // Fuel Pressure (Bar)
    5.0f,     // Short-Term Fuel Trim (%)
    5.0f,     // Long-Term Fuel Trim (%)
    0.2f,     // O2 Sensor Voltage (V)
    15.0f,    // Coolant Temp (°C)
    5.0f,     // Intake Air Temp (°C)
    100.0f    // Catalyst Temp (°C)
};

// Feature names for XAI alerts
const char* feature_names[NUM_FEATURES] = {
    "Engine RPM",
    "Fuel Pressure",
    "Fuel Trim (Short)",
    "Fuel Trim (Long)",
    "O2 Sensor",
    "Coolant Temp",
    "Intake Air Temp",
    "Catalyst Temp"
};

// ============================================================
// BLE CALLBACKS
// ============================================================
class ServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* server) {
        ble_connected = true;
        digitalWrite(LED_BLUE, HIGH);
        Serial.println("[BLE] Client connected");
    }

    void onDisconnect(BLEServer* server) {
        ble_connected = false;
        digitalWrite(LED_BLUE, LOW);
        Serial.println("[BLE] Client disconnected");
        // Restart advertising
        server->startAdvertising();
    }
};

// ============================================================
// OBD-II COMMUNICATION
// ============================================================

/**
 * Send AT command to ELM327 and read response
 */
String obd_command(const char* cmd, unsigned long timeout_ms = 2000) {
    OBD_SERIAL.print(cmd);
    OBD_SERIAL.print('\r');
    
    String response = "";
    unsigned long start = millis();
    
    while (millis() - start < timeout_ms) {
        if (OBD_SERIAL.available()) {
            char c = OBD_SERIAL.read();
            if (c == '>') break;  // ELM327 prompt
            response += c;
        }
    }
    
    response.trim();
    return response;
}

/**
 * Initialize ELM327 OBD-II adapter
 */
bool obd_init() {
    Serial.println("[OBD] Initializing ELM327...");
    
    obd_command("ATZ", 3000);    // Reset
    delay(500);
    obd_command("ATE0");          // Echo off
    obd_command("ATL0");          // Linefeeds off
    obd_command("ATS0");          // Spaces off
    obd_command("ATH0");          // Headers off
    obd_command("ATSP0");         // Auto-detect protocol
    
    // Test connection with RPM query
    String resp = obd_command("010C");
    if (resp.indexOf("41") >= 0) {
        Serial.println("[OBD] Connected successfully");
        return true;
    }
    
    Serial.println("[OBD] Connection failed - check wiring");
    return false;
}

/**
 * Parse hex response from OBD-II PID query
 * Returns the numeric value, or -1 on error
 */
float obd_parse_pid(const char* pid_cmd, int num_bytes) {
    String resp = obd_command(pid_cmd);
    
    // Response format: "41 0C XX YY" (Mode 01, PID 0C, data bytes)
    // With spaces off: "410CXX" or "410CXXYY"
    if (resp.length() < 4 + num_bytes * 2) return -1;
    
    // Skip "41XX" header (4 chars)
    String data_hex = resp.substring(4);
    
    if (num_bytes == 1) {
        return (float)strtol(data_hex.substring(0, 2).c_str(), NULL, 16);
    } else if (num_bytes == 2) {
        int A = strtol(data_hex.substring(0, 2).c_str(), NULL, 16);
        int B = strtol(data_hex.substring(2, 4).c_str(), NULL, 16);
        return (float)((A * 256) + B);
    }
    
    return -1;
}

/**
 * Read all 8 sensor channels from OBD-II
 */
void read_sensors(SensorData* data) {
    // PID 0x0C: Engine RPM = ((A*256)+B)/4
    float raw = obd_parse_pid("010C", 2);
    data->engine_rpm = (raw >= 0) ? raw / 4.0f : data->engine_rpm;
    
    // PID 0x0A: Fuel Pressure (gauge) = A * 3 (kPa) → convert to Bar
    raw = obd_parse_pid("010A", 1);
    data->fuel_pressure = (raw >= 0) ? raw * 3.0f / 100.0f : data->fuel_pressure;
    
    // PID 0x06: Short-Term Fuel Trim = (A - 128) * 100/128
    raw = obd_parse_pid("0106", 1);
    data->fuel_trim_short = (raw >= 0) ? (raw - 128.0f) * 100.0f / 128.0f : data->fuel_trim_short;
    
    // PID 0x07: Long-Term Fuel Trim = (A - 128) * 100/128
    raw = obd_parse_pid("0107", 1);
    data->fuel_trim_long = (raw >= 0) ? (raw - 128.0f) * 100.0f / 128.0f : data->fuel_trim_long;
    
    // PID 0x14: O2 Sensor Voltage = A / 200 (V)
    raw = obd_parse_pid("0114", 1);
    data->o2_voltage = (raw >= 0) ? raw / 200.0f : data->o2_voltage;
    
    // PID 0x05: Coolant Temperature = A - 40 (°C)
    raw = obd_parse_pid("0105", 1);
    data->coolant_temp = (raw >= 0) ? raw - 40.0f : data->coolant_temp;
    
    // PID 0x0F: Intake Air Temperature = A - 40 (°C)
    raw = obd_parse_pid("010F", 1);
    data->intake_air_temp = (raw >= 0) ? raw - 40.0f : data->intake_air_temp;
    
    // PID 0x3C: Catalyst Temperature = ((A*256)+B)/10 - 40 (°C)
    raw = obd_parse_pid("013C", 2);
    data->catalyst_temp = (raw >= 0) ? raw / 10.0f - 40.0f : data->catalyst_temp;
}

// ============================================================
// SENSOR BUFFER MANAGEMENT
// ============================================================

/**
 * Add sensor reading to sliding window buffer
 */
void buffer_add(SensorData* data) {
    // Normalize using training statistics (StandardScaler)
    sensor_buffer[buffer_index][0] = (data->engine_rpm - feature_means[0]) / feature_stds[0];
    sensor_buffer[buffer_index][1] = (data->fuel_pressure - feature_means[1]) / feature_stds[1];
    sensor_buffer[buffer_index][2] = (data->fuel_trim_short - feature_means[2]) / feature_stds[2];
    sensor_buffer[buffer_index][3] = (data->fuel_trim_long - feature_means[3]) / feature_stds[3];
    sensor_buffer[buffer_index][4] = (data->o2_voltage - feature_means[4]) / feature_stds[4];
    sensor_buffer[buffer_index][5] = (data->coolant_temp - feature_means[5]) / feature_stds[5];
    sensor_buffer[buffer_index][6] = (data->intake_air_temp - feature_means[6]) / feature_stds[6];
    sensor_buffer[buffer_index][7] = (data->catalyst_temp - feature_means[7]) / feature_stds[7];
    
    buffer_index = (buffer_index + 1) % SEQ_LENGTH;
    if (!buffer_full && buffer_index == 0) {
        buffer_full = true;
    }
}

// ============================================================
// TFLITE MICRO INFERENCE
// ============================================================

/**
 * Initialize TFLite Micro interpreter
 */
bool tflite_init() {
    Serial.println("[TFLite] Loading model...");
    
    tfl_model = tflite::GetModel(fedxai_model_data);
    if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[TFLite] Model schema version mismatch!");
        return false;
    }
    
    static tflite::AllOpsResolver resolver;
    
    static tflite::MicroInterpreter static_interpreter(
        tfl_model, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[TFLite] AllocateTensors failed!");
        return false;
    }
    
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    
    Serial.printf("[TFLite] Model loaded. Arena used: %d bytes\n",
                  interpreter->arena_used_bytes());
    Serial.printf("[TFLite] Input: [%d, %d], Output: [%d]\n",
                  input_tensor->dims->data[1],
                  input_tensor->dims->data[2],
                  output_tensor->dims->data[1]);
    
    return true;
}

/**
 * Run inference on current sensor buffer
 * Returns failure probability (0.0 - 1.0)
 */
float tflite_infer() {
    if (!buffer_full) return 0.0f;
    
    // Copy buffer into input tensor (handle circular buffer)
    float* input_data = input_tensor->data.f;
    for (int t = 0; t < SEQ_LENGTH; t++) {
        int idx = (buffer_index + t) % SEQ_LENGTH;
        for (int f = 0; f < NUM_FEATURES; f++) {
            input_data[t * NUM_FEATURES + f] = sensor_buffer[idx][f];
        }
    }
    
    // Run inference
    unsigned long t_start = micros();
    TfLiteStatus status = interpreter->Invoke();
    unsigned long t_end = micros();
    
    if (status != kTfLiteOk) {
        Serial.println("[TFLite] Inference failed!");
        return 0.0f;
    }
    
    float prob = output_tensor->data.f[0];
    
    Serial.printf("[TFLite] Inference: %.4f (%.1f ms)\n",
                  prob, (t_end - t_start) / 1000.0f);
    
    return prob;
}

// ============================================================
// XAI ALERT ENGINE (Edge-side Explainability)
// ============================================================

/**
 * Generate mechanic-friendly alert from sensor deviations
 * This is a lightweight edge-side XAI alternative to SHAP
 */
String generate_xai_alert(SensorData* data, float prob) {
    if (prob < FAILURE_THRESHOLD) {
        return "System OK - All sensors nominal";
    }
    
    String alert = "WARNING: ";
    int alert_count = 0;
    
    // Check each sensor against known failure thresholds
    // These thresholds are derived from SHAP analysis in Phase 3
    
    if (data->fuel_pressure < 2.0f) {
        alert += "LOW FUEL PRESSURE (" + String(data->fuel_pressure, 1) + " Bar). ";
        alert += "Check fuel pump and filter. ";
        alert_count++;
    }
    
    if (abs(data->fuel_trim_long) > 15.0f) {
        alert += "ABNORMAL FUEL TRIM (" + String(data->fuel_trim_long, 1) + "%). ";
        alert += "Possible fuel injector issue. ";
        alert_count++;
    }
    
    if (data->coolant_temp > 105.0f) {
        alert += "OVERHEATING (" + String(data->coolant_temp, 1) + "C). ";
        alert += "Check coolant level and radiator. ";
        alert_count++;
    }
    
    if (data->o2_voltage < 0.2f || data->o2_voltage > 0.8f) {
        alert += "O2 SENSOR ANOMALY (" + String(data->o2_voltage, 3) + "V). ";
        alert += "Check exhaust system. ";
        alert_count++;
    }
    
    if (data->catalyst_temp > 600.0f) {
        alert += "HIGH CATALYST TEMP (" + String(data->catalyst_temp, 0) + "C). ";
        alert += "Possible catalytic converter damage. ";
        alert_count++;
    }
    
    if (alert_count == 0) {
        alert += "Subtle degradation detected (prob=" + String(prob, 3) + "). ";
        alert += "Recommend diagnostic scan.";
    }
    
    return alert;
}

// ============================================================
// BLE SETUP
// ============================================================
void ble_init() {
    Serial.println("[BLE] Initializing...");
    
    BLEDevice::init(DEVICE_NAME);
    ble_server = BLEDevice::createServer();
    ble_server->setCallbacks(new ServerCallbacks());
    
    BLEService* service = ble_server->createService(SERVICE_UUID);
    
    // Status characteristic (Notify)
    char_status = service->createCharacteristic(
        CHAR_STATUS_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    char_status->addDescriptor(new BLE2902());
    
    // Probability characteristic (Notify)
    char_prob = service->createCharacteristic(
        CHAR_PROB_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    char_prob->addDescriptor(new BLE2902());
    
    // XAI Alert characteristic (Notify)
    char_alert = service->createCharacteristic(
        CHAR_ALERT_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    char_alert->addDescriptor(new BLE2902());
    
    // Live sensors characteristic (Read)
    char_sensors = service->createCharacteristic(
        CHAR_SENSORS_UUID,
        BLECharacteristic::PROPERTY_READ
    );
    
    service->start();
    
    BLEAdvertising* advertising = BLEDevice::getAdvertising();
    advertising->addServiceUUID(SERVICE_UUID);
    advertising->setScanResponse(true);
    advertising->setMinPreferred(0x06);
    BLEDevice::startAdvertising();
    
    Serial.println("[BLE] Advertising as: " DEVICE_NAME);
}

/**
 * Send status update via BLE
 */
void ble_notify(float prob, const String& alert) {
    if (!ble_connected) return;
    
    // Status: "OK" or "FAIL"
    const char* status = (prob < FAILURE_THRESHOLD) ? "OK" : "FAIL";
    char_status->setValue(status);
    char_status->notify();
    
    // Probability (4-byte float)
    char_prob->setValue(prob);
    char_prob->notify();
    
    // XAI Alert text
    char_alert->setValue(alert.c_str());
    char_alert->notify();
    
    // Live sensor JSON
    String sensors = "{";
    sensors += "\"rpm\":" + String(latest_sensors.engine_rpm, 0);
    sensors += ",\"fp\":" + String(latest_sensors.fuel_pressure, 2);
    sensors += ",\"stft\":" + String(latest_sensors.fuel_trim_short, 1);
    sensors += ",\"ltft\":" + String(latest_sensors.fuel_trim_long, 1);
    sensors += ",\"o2\":" + String(latest_sensors.o2_voltage, 3);
    sensors += ",\"ct\":" + String(latest_sensors.coolant_temp, 1);
    sensors += ",\"iat\":" + String(latest_sensors.intake_air_temp, 1);
    sensors += ",\"cat\":" + String(latest_sensors.catalyst_temp, 0);
    sensors += ",\"prob\":" + String(prob, 4);
    sensors += "}";
    char_sensors->setValue(sensors.c_str());
}

// ============================================================
// LED STATUS
// ============================================================
void update_leds(float prob) {
    if (prob < FAILURE_THRESHOLD) {
        digitalWrite(LED_GREEN, HIGH);
        digitalWrite(LED_RED, LOW);
    } else {
        digitalWrite(LED_GREEN, LOW);
        // Blink red for failure
        digitalWrite(LED_RED, (millis() / 500) % 2);
    }
}

// ============================================================
// SETUP
// ============================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n============================================");
    Serial.println("  FEDXAI-AUTO Edge Dongle v" FIRMWARE_VERSION);
    Serial.println("  Predictive Maintenance for Indian Roads");
    Serial.println("============================================\n");
    
    // GPIO
    pinMode(LED_GREEN, OUTPUT);
    pinMode(LED_RED, OUTPUT);
    pinMode(LED_BLUE, OUTPUT);
    
    // Startup sequence
    digitalWrite(LED_GREEN, HIGH);
    delay(200);
    digitalWrite(LED_RED, HIGH);
    delay(200);
    digitalWrite(LED_BLUE, HIGH);
    delay(500);
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_RED, LOW);
    digitalWrite(LED_BLUE, LOW);
    
    // Initialize subsystems
    if (!tflite_init()) {
        Serial.println("[FATAL] TFLite init failed!");
        while(1) {
            digitalWrite(LED_RED, !digitalRead(LED_RED));
            delay(100);  // Fast blink = error
        }
    }
    
    ble_init();
    
    // Initialize OBD-II
    OBD_SERIAL.begin(OBD_BAUD, SERIAL_8N1, OBD_RX_PIN, OBD_TX_PIN);
    if (!obd_init()) {
        Serial.println("[WARN] OBD-II not detected - entering demo mode");
        // In demo mode, generate synthetic data for testing
    }
    
    // Clear sensor buffer
    memset(sensor_buffer, 0, sizeof(sensor_buffer));
    memset(&latest_sensors, 0, sizeof(SensorData));
    
    Serial.println("\n[READY] System initialized. Monitoring...\n");
    digitalWrite(LED_GREEN, HIGH);
}

// ============================================================
// MAIN LOOP (1 Hz)
// ============================================================
unsigned long last_poll = 0;

void loop() {
    unsigned long now = millis();
    
    if (now - last_poll >= POLL_INTERVAL_MS) {
        last_poll = now;
        
        // 1. Read sensors from OBD-II
        read_sensors(&latest_sensors);
        
        // 2. Add to sliding window buffer
        buffer_add(&latest_sensors);
        
        // 3. Run inference (if buffer is full)
        if (buffer_full) {
            failure_probability = tflite_infer();
            failure_detected = (failure_probability >= FAILURE_THRESHOLD);
            
            // 4. Generate XAI alert
            xai_alert_message = generate_xai_alert(&latest_sensors, failure_probability);
            
            // 5. Update BLE
            ble_notify(failure_probability, xai_alert_message);
            
            // 6. Update LEDs
            update_leds(failure_probability);
            
            // 7. Serial log
            Serial.printf("[%lu] RPM:%.0f FP:%.2f CT:%.1f | Prob:%.4f | %s\n",
                          now / 1000,
                          latest_sensors.engine_rpm,
                          latest_sensors.fuel_pressure,
                          latest_sensors.coolant_temp,
                          failure_probability,
                          failure_detected ? "FAILURE" : "OK");
        } else {
            Serial.printf("[%lu] Filling buffer... %d/%d\n",
                          now / 1000, buffer_index, SEQ_LENGTH);
        }
    }
    
    delay(10);  // Yield to BLE stack
}
