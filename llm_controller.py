def llm_decision(gesture):
    mapping = {
        "MOVE_UP": "arm_up",
        "MOVE_DOWN": "arm_down",
        "MOVE_LEFT": "rotate_left",
        "MOVE_RIGHT": "rotate_right",
        "STOP": "stop",
        "OPEN": "place",
        "CLOSE": "pick"
    }
    return mapping.get(gesture, "stop")