COLUMN_GROUPS = {
    "base": [],
    "acceleration": ["ACCELERATION",
                     "ACCELERATION_PEDAL",
                     "ACCELERATION_Y",
                     "ACCELERATION_Z"],
    "speed": ["SPEED",
              "SPEED_NEXT_VEHICLE",
              "SPEED_Y",
              "SPEED_Z"],
    "distance": ["DISTANCE",
                 "DISTANCE_TO_NEXT_INTERSECTION",
                 "DISTANCE_TO_NEXT_STOP_SIGNAL",
                 "DISTANCE_TO_NEXT_TRAFFIC_LIGHT_SIGNAL",
                 "DISTANCE_TO_NEXT_VEHICLE",
                 "DISTANCE_TO_NEXT_YIELD_SIGNAL"],
    "pedal": ["BRAKE_PEDAL", "CLUTCH_PEDAL"],
    "lane": ["LANE",
             "LANE_LATERAL_SHIFT_CENTER",
             "LANE_LATERAL_SHIFT_LEFT",
             "LANE_LATERAL_SHIFT_RIGHT",
             "LANE_WIDTH",
             "FAST_LANE"],
    "rainsnow": ["RAIN", "REAR_WIPERS", "FRONT_WIPERS", "SNOW"],
    "angle": ["CURVE_RADIUS", "ROAD_ANGLE", "STEERING_WHEEL"],
    "indicator": ["INDICATORS", "INDICATORS_ON_INTERSECTION"],
    "headlight": ["HEAD_LIGHTS"],
    "horn": ["HORN"],
    "gearbox": ["GEARBOX"]
}

COLUMN_SELECTION_SPECS = ["base", "acceleration", "speed", "distance", "pedal",
                          "lane", "rainsnow", "angle",
                          "indicator", "headlight", "horn", "gearbox"]

INPUT_LENGTH = 300
CHANNELS_SIZES = [25, 32]
OUTPUT_SIZE = 62
KERNEL_SIZE = 16
DROPOUT = 0.1

LR = 0.001
WEIGHT_DECAY = 0.0
BATCH_SIZE = 30
EPOCHS = 1
