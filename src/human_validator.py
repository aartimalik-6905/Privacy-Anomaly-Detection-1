def is_human_skeleton(landmarks):
    # Required landmarks (human anatomy)
    try:
        ls = landmarks[11]  # left shoulder
        rs = landmarks[12]  # right shoulder
        lh = landmarks[23]  # left hip
        rh = landmarks[24]  # right hip
        la = landmarks[27]  # left ankle
        ra = landmarks[28]  # right ankle
    except:
        return False

    # Shoulder width
    shoulder_width = abs(ls.x - rs.x)

    # Torso height
    torso_height = abs(((ls.y + rs.y)/2) - ((lh.y + rh.y)/2))

    # Leg height
    leg_height = abs(((lh.y + rh.y)/2) - ((la.y + ra.y)/2))

    # Human anatomical ratios
    if shoulder_width < 0.08:
        return False

    if not (0.4 < leg_height / torso_height < 1.2):
        return False

    return True
