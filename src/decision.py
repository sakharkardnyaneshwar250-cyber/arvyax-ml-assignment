def decide_action(state, intensity, stress, energy, time_of_day):

    if stress > 4 and energy < 3:
        return "box_breathing", "now"

    if state == "sad" and intensity >= 4:
        return "journaling", "within_15_min"

    if energy >= 4 and stress <= 2:
        return "deep_work", "now"

    if energy <= 2:
        return "rest", "later_today"

    if time_of_day == "night":
        return "sleep", "tonight"

    return "light_planning", "within_15_min"
