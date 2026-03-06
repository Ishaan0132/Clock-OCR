import math
from .geometry_utils import get_vector, dot_product, cross_product

def get_angle(hand, center_x, center_y):
    u = get_vector(hand)
    # Reference vector pointing up (12 o'clock)
    v = [center_x - center_x, center_y - (center_y - 100)]

    dot_uv = dot_product(u, v)
    length_u = math.sqrt(u[0]**2 + u[1]**2)
    length_v = math.sqrt(v[0]**2 + v[1]**2)

    if length_u == 0: return 0

    cos_theta = dot_uv / (length_u * length_v)
    cos_theta = max(min(cos_theta, 1.0), -1.0)

    theta = math.acos(cos_theta)
    theta_degrees = math.degrees(theta)

    cross_uv = cross_product(u, v)
    if cross_uv > 0:
        return 360 - theta_degrees
    else:
        return theta_degrees

def get_time(hour_angle, minute_angle, second_angle):
    hour = hour_angle / 30
    minute = minute_angle / 6
    second = second_angle / 6

    # Edge case corrections
    if (round(hour)*30 - hour_angle <= 6) and ((355 < minute_angle < 360) or (minute_angle < 90)):
        hour = round(hour)
        if hour == 12: hour = 0

    if (hour_angle - hour*30 <= 6) and (355 < minute_angle < 360):
        minute = 0

    if (round(minute)*6 - minute_angle <= 6) and (second_angle < 6):
        minute = round(minute)
        if minute == 60: minute = 0

    if (minute_angle - minute*30 <= 6) and (354 < second_angle < 360):
        second = 0

    hour = int(hour)
    minute = int(minute)
    second = int(second)

    return f"{hour:02d}:{minute:02d}:{second:02d}"