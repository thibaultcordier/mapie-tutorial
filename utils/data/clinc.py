import numpy as np


NB_CLASSES = 151
FEATURES = 'text'
LABEL = 'intent_name'
TARGET = 'intent'
PREDICTION = 'prediction'
SET = 'set'
SET_LABEL = 'set_label'


def sample_and_split_dataset(ds, nb_train=1000, nb_calib=1000, nb_test=1000):
    np.random.seed(10)

    nb_train = min(nb_train, len(ds['train']) - NB_CLASSES)
    df_train = ds['train'].train_test_split(
        train_size=nb_train, test_size=NB_CLASSES, stratify_by_column=TARGET
    )['train']

    nb_calib = min(nb_calib, len(ds['validation']) - NB_CLASSES)
    df_calib = ds['validation'].train_test_split(
        train_size=nb_calib, test_size=NB_CLASSES, stratify_by_column=TARGET
    )['train']

    nb_test = min(nb_test, len(ds['test']) - NB_CLASSES)
    df_test = ds['test'].train_test_split(
        train_size=nb_test, test_size=NB_CLASSES, stratify_by_column=TARGET
    )['train']
    
    return df_train, df_calib, df_test


def get_dataframe(*ds_list):
    for ds in ds_list:
        yield ds


def get_X_y(*df_list):
    for df in df_list:
        X, y = df[FEATURES], df[TARGET]
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        yield X, y


def get_label_mapping(ds):
    idx2lab = {i: ds.features[TARGET].int2str(i) for i in range(NB_CLASSES)}
    lab2idx = {ds.features[TARGET].int2str(i): i for i in range(NB_CLASSES)}
    label_list = sorted(lab2idx, key=lambda name: lab2idx[name])
    return idx2lab, lab2idx, label_list


dom2acts = {
    "banking": [
        "freeze_account",
        "routing",
        "pin_change",
        "bill_due",
        "pay_bill",
        "account_blocked",
        "interest_rate",
        "min_payment",
        "bill_balance",
        "transfer",
        "order_checks",
        "balance",
        "spending_history",
        "transactions",
        "report_fraud"
    ],
    "credit_cards": [
        "replacement_card_duration",
        "expiration_date",
        "damaged_card",
        "improve_credit_score",
        "report_lost_card",
        "card_declined",
        "credit_limit_change",
        "apr",
        "redeem_rewards",
        "credit_limit",
        "rewards_balance",
        "application_status",
        "credit_score",
        "new_card",
        "international_fees"
    ],
    "kitchen_and_dining": [
        "food_last",
        "confirm_reservation",
        "how_busy",
        "ingredients_list",
        "calories",
        "nutrition_info",
        "recipe",
        "restaurant_reviews",
        "restaurant_reservation",
        "meal_suggestion",
        "restaurant_suggestion",
        "cancel_reservation",
        "ingredient_substitution",
        "cook_time",
        "accept_reservations"
    ],
    "home": [
        "what_song",
        "play_music",
        "todo_list_update",
        "reminder",
        "reminder_update",
        "calendar_update",
        "order_status",
        "update_playlist",
        "shopping_list",
        "calendar",
        "next_song",
        "order",
        "todo_list",
        "shopping_list_update",
        "smart_home"
    ],
    "auto_and_commute": [
        "current_location",
        "oil_change_when",
        "oil_change_how",
        "uber",
        "traffic",
        "tire_pressure",
        "schedule_maintenance",
        "gas",
        "mpg",
        "distance",
        "directions",
        "last_maintenance",
        "gas_type",
        "tire_change",
        "jump_start"
    ],
    "travel": [
        "plug_type",
        "travel_notification",
        "translate",
        "flight_status",
        "international_visa",
        "timezone",
        "exchange_rate",
        "travel_suggestion",
        "travel_alert",
        "vaccines",
        "lost_luggage",
        "book_flight",
        "book_hotel",
        "carry_on",
        "car_rental"
    ],
    "utility": [
        "weather",
        "alarm",
        "date",
        "find_phone",
        "share_location",
        "timer",
        "make_call",
        "calculator",
        "definition",
        "measurement_conversion",
        "flip_coin",
        "spelling",
        "time",
        "roll_dice",
        "text"
    ],
    "work": [
        "pto_request_status",
        "next_holiday",
        "insurance_change",
        "insurance",
        "meeting_schedule",
        "payday",
        "taxes",
        "income",
        "rollover_401k",
        "pto_balance",
        "pto_request",
        "w2",
        "schedule_meeting",
        "direct_deposit",
        "pto_used"
    ],
    "small_talk": [
        "who_made_you",
        "meaning_of_life",
        "who_do_you_work_for",
        "do_you_have_pets",
        "what_are_your_hobbies",
        "fun_fact",
        "what_is_your_name",
        "where_are_you_from",
        "goodbye",
        "thank_you",
        "greeting",
        "tell_joke",
        "are_you_a_bot",
        "how_old_are_you",
        "what_can_i_ask_you"
    ],
    "meta": [
        "change_speed",
        "user_name",
        "whisper_mode",
        "yes",
        "change_volume",
        "no",
        "change_language",
        "repeat",
        "change_accent",
        "cancel",
        "sync_device",
        "change_user_name",
        "change_ai_name",
        "reset_settings",
        "maybe"
    ],
    "oos": ["oos"]
}

act2dom = {
    act: dom for dom in dom2acts for act in dom2acts[dom]
}

dom2idx = {dom: idx for idx, dom in enumerate(dom2acts)}
idx2dom = {idx: dom for idx, dom in enumerate(dom2acts)}
dom_list = list(dom2acts.keys())
