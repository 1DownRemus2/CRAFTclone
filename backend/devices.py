# devices.py
DEVICE_REGISTRY = [
    {
        "id": "hub_zigbee",
        "description": "Smart hub with Zigbee and WiFi connectivity for home automation",
        "provides": {"zigbee_network", "wifi_network"},
        "requires": set(),
        "metrics": {"power": 5, "cost": 80},
        "tags": ["iot", "automation", "connectivity"]
    },
    {
        "id": "bulb_zigbee",
        "description": "Low power Zigbee smart bulb for adjustable home lighting",
        "provides": {"lighting"},
        "requires": {"zigbee_network"},
        "metrics": {"power": 7, "cost": 15},
        "tags": ["lighting", "energy", "smart bulb"]
    },
    {
        "id": "voice_speaker",
        "description": "Voice assistant smart speaker for hands-free voice control",
        "provides": {"voice_control"},
        "requires": {"wifi_network"},
        "metrics": {"power": 8, "cost": 50},
        "tags": ["voice", "assistant", "smart speaker"]
    },
    {
        "id": "motion_sensor",
        "description": "Wireless motion sensor for detecting movement and presence",
        "provides": {"motion_detection"},
        "requires": {"zigbee_network"},
        "metrics": {"power": 3, "cost": 25},
        "tags": ["sensor", "motion"]
    },
    {
        "id": "desk",
        "description": "Ergonomic office desk providing workspace for productivity",
        "provides": {"workspace"},
        "requires": set(),
        "metrics": {"power": 0, "cost": 120},
        "tags": ["office", "furniture"]
    },
    {
        "id": "monitor",
        "description": "High resolution computer monitor for visual display output",
        "provides": {"visual_output"},
        "requires": {"video_output"},
        "metrics": {"power": 40, "cost": 250},
        "tags": ["display", "screen"]
    },
    {
        "id": "computer",
        "description": "Desktop computer providing processing power and video output",
        "provides": {"compute", "video_output"},
        "requires": set(),
        "metrics": {"power": 150, "cost": 600},
        "tags": ["computer", "PC"]
    },
    # New devices not explicitly tagged with capabilities
    {
        "id": "alexa_echo",
        "description": "Amazon Echo device with Alexa voice assistant for smart home control",
        "provides": set(),  # Empty - will be inferred!
        "requires": {"wifi_network"},
        "metrics": {"power": 10, "cost": 60},
        "tags": ["amazon", "assistant"]
    },
    {
        "id": "led_strip",
        "description": "RGB LED light strip for ambient room illumination",
        "provides": set(),  # Empty - will be inferred!
        "requires": {"wifi_network"},
        "metrics": {"power": 12, "cost": 30},
        "tags": ["led", "rgb"]
    },
    # Additional test devices with NO explicit capabilities
    {
        "id": "homepod_mini",
        "description": "Apple HomePod mini with Siri voice assistant and smart home integration",
        "provides": set(),  # Empty - should infer voice_control!
        "requires": {"wifi_network"},
        "metrics": {"power": 9, "cost": 99},
        "tags": ["apple", "speaker"]
    },
    {
        "id": "philips_hue",
        "description": "Philips Hue smart bulb with adjustable brightness and color temperature",
        "provides": set(),  # Empty - should infer lighting!
        "requires": {"zigbee_network"},
        "metrics": {"power": 8, "cost": 20},
        "tags": ["philips", "hue"]
    },
    {
        "id": "aqara_sensor",
        "description": "Aqara wireless sensor detecting motion and human presence in rooms",
        "provides": set(),  # Empty - should infer motion_detection!
        "requires": {"zigbee_network"},
        "metrics": {"power": 2, "cost": 22},
        "tags": ["aqara", "wireless"]
    },
    {
        "id": "standing_desk",
        "description": "Electric standing desk with adjustable height for ergonomic workspace",
        "provides": set(),  # Empty - should infer workspace!
        "requires": set(),
        "metrics": {"power": 5, "cost": 450},
        "tags": ["furniture", "adjustable"]
    },
    {
        "id": "ultrawide_monitor",
        "description": "34-inch ultrawide curved monitor for immersive visual display",
        "provides": set(),  # Empty - should infer visual_output!
        "requires": {"video_output"},
        "metrics": {"power": 50, "cost": 400},
        "tags": ["ultrawide", "curved"]
    },
    {
        "id": "laptop_workstation",
        "description": "High performance laptop with powerful processor for computing tasks",
        "provides": set(),  # Empty - should infer compute + video_output!
        "requires": set(),
        "metrics": {"power": 65, "cost": 1200},
        "tags": ["laptop", "portable"]
    },
]