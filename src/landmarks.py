"""Configuração, colunas e extração compartilhadas pela coleta e pelo app."""

import mediapipe as mp

mp_holistic = mp.solutions.holistic
Pose = mp_holistic.PoseLandmark

LANDMARK_CONFIG = {
    "pose": {
        "total": 33,
        "indices": [
            #Pose.NOSE.value,
            #Pose.LEFT_EYE.value,
            #Pose.RIGHT_EYE.value,
            #Pose.MOUTH_LEFT.value,
            #Pose.MOUTH_RIGHT.value,
            Pose.LEFT_SHOULDER.value,
            Pose.RIGHT_SHOULDER.value,
            Pose.LEFT_ELBOW.value,
            Pose.RIGHT_ELBOW.value,
            Pose.LEFT_WRIST.value,
            Pose.RIGHT_WRIST.value,
        ],
        "include_visibility": True,
    },
    "left_hand": {
        "total": 21,
        "indices": list(range(21)),
        "include_visibility": False,
    },
    "right_hand": {
        "total": 21,
        "indices": list(range(21)),
        "include_visibility": False,
    },
    "face": {
        "total": 468,
        "indices": [
            # Nariz
            1, 2, 4, 5, 6, 19,

            # Olho direito
            33, 133, 159, 145, 163, 7,

            # Olho esquerdo
            263, 362, 386, 374, 390, 249,

            # Sobrancelha direita
            46, 53, 52, 65, 55, 70, 63, 105,

            # Sobrancelha esquerda
            276, 283, 282, 295, 285, 300, 293, 334,

            # Boca
            61, 291,
            0, 17,
            13, 14,
            78, 308,
            87, 317
        ],
        "include_visibility": False,
    },
}

RESULT_ATTR = {
    "pose": "pose_landmarks",
    "face": "face_landmarks",
    "left_hand": "left_hand_landmarks",
    "right_hand": "right_hand_landmarks",
}

def build_column_names(config: dict) -> list:
    columns = []
    for group_name, group_cfg in config.items():
        for idx in group_cfg["indices"]:
            columns.extend([
                f"{group_name}_{idx}_x",
                f"{group_name}_{idx}_y",
                f"{group_name}_{idx}_z",
            ])
            if group_cfg["include_visibility"]:
                columns.append(f"{group_name}_{idx}_v")
    columns.append("participant_id")
    columns.append("label")
    return columns


def build_feature_column_names(config: dict) -> list:
    """
    Igual a build_column_names, mas SEM participant_id/label — é essa lista
    que a normalização usa, porque ela precisa saber, posição a posição em
    'row', quais entradas são x/y/z/v (participant_id e label não entram
    nessa matemática).
    """
    return build_column_names(config)[:-2]


def extract_group(landmark_list, group_cfg: dict) -> list:
    values = []
    n_coords = 4 if group_cfg["include_visibility"] else 3

    for idx in group_cfg["indices"]:
        if landmark_list is not None:
            lm = landmark_list.landmark[idx]
            point = [lm.x, lm.y, lm.z]
            if group_cfg["include_visibility"]:
                point.append(lm.visibility)
            values.extend(point)
        else:
            values.extend([0.0] * n_coords)

    return values


def extract_landmarks(results, config: dict = LANDMARK_CONFIG) -> list:
    row = []
    for group_name, group_cfg in config.items():
        attr_name = RESULT_ATTR[group_name]
        landmark_list = getattr(results, attr_name)
        row.extend(extract_group(landmark_list, group_cfg))
    return row