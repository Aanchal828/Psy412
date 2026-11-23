"""
PSY412 / Cyberball + Larson & Perry-style visual-capture task
PsychoPy + Tobii Pro Nano (tobii_research SDK)

IMPORTANT DESIGN NOTES
---------------------
This is an ADAPTED replication, not an exact reproduction.
- Preserves the original within-trial sequence:
  central fixation -> central bracket discrimination (until response) ->
  peripheral cue (300 ms) -> peripheral bracket target (100 ms) ->
  second bracket discrimination response.
- Prosaccade: cue and peripheral target are on the same side.
- Antisaccade: cue and peripheral target are on opposite sides.
- 20 trials per block; 2 blocks per stage; pre and post Cyberball = 80 test trials.
- Four brief practice trials are given once before the pre-test (2 per task).
- Questionnaires and Cyberball are run in the same PsychoPy window.
- Condition and task order are assigned in balanced randomised blocks of four.
- Calibration is performed once before pre-test; a short validation is repeated
  before post-test. Cyberball and questionnaires are integrated into this script.

The script records raw gaze samples, event timestamps, and trial-level responses.
It DOES NOT automatically classify saccades/capture errors. Those must be derived
from the raw gaze stream using a pilot-validated detector and event timestamps.

Run in PsychoPy Coder. Install tobii_research in the same Python environment.
Edit CONFIG before data collection. Test with the actual monitor and tracker.
"""

from psychopy import visual, core, gui, monitors, logging
from psychopy.hardware import keyboard
from psychopy import event
import tobii_research as tr
import csv
import os
import math
import random
import threading
from datetime import datetime

# ================================ CONFIG ================================
MONITOR_NAME = 'psy412_tobii'   # own profile: mon.save() below would overwrite 'testMonitor' for other lab users
SCREEN_RES = (1366, 768)         # real display resolution (confirmed)
SCREEN_WIDTH_CM = 34.5          # CONFIRM with a ruler (visible picture, edge to edge)
VIEW_DIST_CM = 71.7             # CONFIRM: eyes to screen, participant seated in position
FULLSCR = True
BACKGROUND = (-1, -1, -1)

# The script prints the real edge margin at start-up. With 34.5 cm at 71.7 cm,
# +/-12 deg is about 15 cm from centre, leaving about 2 cm to the screen edge.
ECCENTRICITY_DEG = 12.0
CUE_SIZE_DEG = 0.8           # Approximate 1 cm square; verify on your display.
BRACKET_HEIGHT_DEG = 1.6
FIXATION_HEIGHT_DEG = 1.2

CUE_DUR = 0.300              # Original paper: 300 ms
PERIPHERAL_TARGET_DUR = 0.100  # Original paper: 100 ms
FIXATION_MIN = 0.500         # Adapted random foreperiod; pilot this.
FIXATION_MAX = 1.000
CENTRAL_RESPONSE_TIMEOUT = 4.0  # Practical safety timeout; no response = omission.
PERIPHERAL_RESPONSE_TIMEOUT = 3.0

N_TEST_TRIALS_PER_BLOCK = 20
N_PRACTICE_PER_TASK = 2      # Two practice trials per task, pre-test only.
LEFT_RESPONSE_KEY = 'left'
RIGHT_RESPONSE_KEY = 'right'
CONTINUE_KEY = 'space'
QUIT_KEY = 'escape'
CHECK_AFTER_CYBERBALL = True   # True: 5-dot check (about 30 s) before the post blocks.
                                # False: go straight into the post blocks. Gaze is recorded
                                # throughout; check accuracy afterwards from the central fixation.

# Tobii Pro SDK Python: True/1 = valid; False/0 = invalid.
# ========================================================================

logging.console.setLevel(logging.WARNING)
random.seed()

# ---------- participant / balanced allocation ----------
exp_info = {'participant': ''}
dlg = gui.DlgFromDict(exp_info, title='PSY412 Integrated Experiment',
                      order=['participant'])
if not dlg.OK or not exp_info['participant'].strip():
    core.quit()
participant = exp_info['participant'].strip()

# Persistent, randomised complete-block allocation across the four combinations.
# Each four-participant block contains each combination exactly once.
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
allocation_path = os.path.join(data_dir, 'allocation_schedule.csv')

def get_allocation(pid):
    rows = []
    if os.path.exists(allocation_path):
        with open(allocation_path, 'r', newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    for r in rows:
        if r['participant'] == pid:
            return r['condition'], r['order']
    used = len(rows)
    if used % 4 == 0:
        block = [
            {'condition': 'included', 'order': 'pro_anti'},
            {'condition': 'excluded', 'order': 'pro_anti'},
            {'condition': 'included', 'order': 'anti_pro'},
            {'condition': 'excluded', 'order': 'anti_pro'},
        ]
        random.shuffle(block)
        pending = block
    else:
        # Reconstruct the remainder of the current four-person block from saved rows.
        combos = [
            {'condition': 'included', 'order': 'pro_anti'},
            {'condition': 'excluded', 'order': 'pro_anti'},
            {'condition': 'included', 'order': 'anti_pro'},
            {'condition': 'excluded', 'order': 'anti_pro'},
        ]
        already = [(r['condition'], r['order']) for r in rows[-(used % 4):]]
        remaining = [c for c in combos if (c['condition'], c['order']) not in already]
        random.shuffle(remaining)
        pending = remaining
    assignment = pending[0]
    with open(allocation_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['participant', 'condition', 'order', 'assigned_at'])
        if os.path.getsize(allocation_path) == 0:
            w.writeheader()
        w.writerow({'participant': pid, 'condition': assignment['condition'],
                    'order': assignment['order'], 'assigned_at': datetime.now().isoformat(timespec='seconds')})
    return assignment['condition'], assignment['order']

condition, order = get_allocation(participant)
print('Allocation:', participant, condition, order)

# ---------- output files ----------
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
stem = os.path.join(data_dir, '{}_{}_{}_{}'.format(participant, condition, order, stamp))

trial_path = stem + '_trials.csv'
gaze_path = stem + '_gaze.csv'
event_path = stem + '_events.csv'
cal_path = stem + '_calibration.csv'
debug_gaze_path = stem + '_gaze_debug.csv'

trial_fields = [
    'participant', 'condition', 'order', 'stage', 'block_number', 'task',
    'trial_number', 'practice', 'cue_side', 'target_side',
    'central_bracket', 'central_response', 'central_correct', 'central_rt_s',
    'peripheral_bracket', 'peripheral_response', 'peripheral_correct',
    'peripheral_rt_s', 'central_omission', 'peripheral_omission',
    'fixation_foreperiod_s', 'cue_onset_system_us', 'cue_offset_system_us',
    'peripheral_target_onset_system_us', 'peripheral_target_offset_system_us'
]
trial_file = open(trial_path, 'w', newline='', encoding='utf-8')
trial_writer = csv.DictWriter(trial_file, fieldnames=trial_fields)
trial_writer.writeheader()

gaze_file = open(gaze_path, 'w', newline='', encoding='utf-8')
gaze_writer = csv.writer(gaze_file)
gaze_writer.writerow([
    'system_time_us', 'device_time_us',
    'left_x_norm', 'left_y_norm', 'left_validity',
    'right_x_norm', 'right_y_norm', 'right_validity'
])

event_file = open(event_path, 'w', newline='', encoding='utf-8')
event_fields = ['system_time_us', 'event', 'participant', 'condition',
                'order', 'stage', 'block_number', 'task', 'trial_number', 'practice']
event_writer = csv.DictWriter(event_file, fieldnames=event_fields)
event_writer.writeheader()

cal_file = open(cal_path, 'w', newline='', encoding='utf-8')
cal_writer = csv.writer(cal_file)
cal_writer.writerow(['stage', 'point', 'target_x_norm', 'target_y_norm',
                     'gaze_x_norm_median', 'gaze_y_norm_median',
                     'error_deg', 'n_valid_samples'])

debug_gaze_file = open(debug_gaze_path, 'w', newline='', encoding='utf-8')
debug_gaze_writer = csv.writer(debug_gaze_file)
debug_gaze_writer.writerow([
    'system_time_us', 'device_time_us',
    'left_point_repr', 'left_point_type', 'left_validity_repr',
    'right_point_repr', 'right_point_type', 'right_validity_repr',
    'left_point_is_finite_pair', 'right_point_is_finite_pair'
])

# ---------- monitor / window ----------
mon = monitors.Monitor(MONITOR_NAME)
mon.setWidth(SCREEN_WIDTH_CM)
mon.setDistance(VIEW_DIST_CM)
mon.setSizePix(SCREEN_RES)
mon.save()

win = visual.Window(size=SCREEN_RES, fullscr=FULLSCR, monitor=mon,
                    units='deg', color=BACKGROUND, allowGUI=False,
                    waitBlanking=True)
fps = win.getActualFrameRate(nIdentical=20, nMaxFrames=120, nWarmUpFrames=20) or 60.0
print('Detected refresh rate: {:.2f} Hz'.format(fps))
print('Cue/target timing will be frame-quantised; inspect actual frame intervals in pilot.')

kb = keyboard.Keyboard()
fix = visual.TextStim(win, text='*', height=FIXATION_HEIGHT_DEG, color='white')
central_bracket_stim = visual.TextStim(win, text='{', height=BRACKET_HEIGHT_DEG, color='white')
peripheral_bracket_stim = visual.TextStim(win, text='{', height=BRACKET_HEIGHT_DEG, color='white')
cue_stim = visual.Rect(win, width=CUE_SIZE_DEG, height=CUE_SIZE_DEG,
                       lineColor='white', fillColor=None, lineWidth=2)
message = visual.TextStim(win, text='', height=0.75, color='white', wrapWidth=28,
                          alignText='center')

# ---------- Tobii recording ----------
CM_PER_DEG = VIEW_DIST_CM * math.tan(math.radians(1.0))
SCREEN_HEIGHT_CM = SCREEN_WIDTH_CM * SCREEN_RES[1] / SCREEN_RES[0]

# Map Tobii normalized display coordinates (origin top-left) to PsychoPy
# degree units using the confirmed physical display geometry. This is the
# same geometry used to position the calibration/validation targets.
def norm_to_deg(xn, yn):
    return ((xn - 0.5) * SCREEN_WIDTH_CM / CM_PER_DEG,
            (0.5 - yn) * SCREEN_HEIGHT_CM / CM_PER_DEG)


def deg_to_norm(xd, yd):
    return (0.5 + (xd * CM_PER_DEG / SCREEN_WIDTH_CM),
            0.5 - (yd * CM_PER_DEG / SCREEN_HEIGHT_CM))


def valid_gaze_xy(gaze_dict, eye):
    point = gaze_dict.get('{}_gaze_point_on_display_area'.format(eye))
    validity = gaze_dict.get('{}_gaze_point_validity'.format(eye))
    # Tobii can occasionally provide non-finite coordinates. Do not treat NaN
    # or infinity as a valid gaze sample, even if the validity flag is 1.
    if point is not None and validity == 1:
        try:
            x, y = float(point[0]), float(point[1])
            if math.isfinite(x) and math.isfinite(y):
                return x, y
        except (TypeError, ValueError, IndexError):
            pass
    return None, None


class TobiiRecorder:
    def __init__(self):
        trackers = tr.find_all_eyetrackers()
        if not trackers:
            raise RuntimeError('No Tobii eye tracker found. Check USB connection and Tobii Pro Eye Tracker Manager.')
        self.tracker = trackers[0]
        self.rows = []
        self.debug_rows = []
        self.lock = threading.Lock()
        self.latest = None
        print('Connected tracker:', self.tracker.model, self.tracker.serial_number)

    def callback(self, g):
        # Preserve the SDK's unprocessed values in a separate diagnostic file.
        # This lets us distinguish missing fields, invalid gaze, and NaN values.
        left_raw = g.get('left_gaze_point_on_display_area')
        right_raw = g.get('right_gaze_point_on_display_area')
        left_validity = g.get('left_gaze_point_validity')
        right_validity = g.get('right_gaze_point_validity')

        def finite_pair(point):
            try:
                return (point is not None and len(point) >= 2
                        and math.isfinite(float(point[0]))
                        and math.isfinite(float(point[1])))
            except (TypeError, ValueError, IndexError):
                return False

        lx, ly = valid_gaze_xy(g, 'left')
        rx, ry = valid_gaze_xy(g, 'right')
        system_ts = g.get('system_time_stamp')
        device_ts = g.get('device_time_stamp')
        row = (system_ts, device_ts, lx, ly, left_validity,
               rx, ry, right_validity)
        debug_row = [
            system_ts, device_ts,
            repr(left_raw), type(left_raw).__name__, repr(left_validity),
            repr(right_raw), type(right_raw).__name__, repr(right_validity),
            finite_pair(left_raw), finite_pair(right_raw)
        ]
        with self.lock:
            self.rows.append(row)
            self.latest = row
            # The callback only queues diagnostic rows; disk writes happen in
            # the main thread to avoid blocking the SDK callback.
            self.debug_rows.append(debug_row)

    def start(self):
        self.tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self.callback,
                                  as_dictionary=True)

    def snapshot(self):
        with self.lock:
            return list(self.rows)

    def debug_snapshot(self):
        with self.lock:
            return list(self.debug_rows)

    def clear(self):
        with self.lock:
            self.rows.clear()
            self.debug_rows.clear()

    def stop(self):
        self.tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self.callback)


rec = TobiiRecorder()
print('Raw gaze diagnostics will be saved to:', debug_gaze_path)

# ---------- event markers / utility ----------
CUR = {'stage': '', 'block_number': '', 'task': '', 'trial_number': '', 'practice': 0}


def mark(event):
    row = {
        'system_time_us': tr.get_system_time_stamp(), 'event': event,
        'participant': participant, 'condition': condition, 'order': order
    }
    row.update(CUR)
    event_writer.writerow(row)
    event_file.flush()
    return row['system_time_us']


def flush_gaze():
    with rec.lock:                      # take the rows and remove exactly those, atomically
        rows = list(rec.rows)
        debug_rows = list(rec.debug_rows)
        del rec.rows[:len(rows)]
        del rec.debug_rows[:len(debug_rows)]
    if rows:
        gaze_writer.writerows(rows)
        gaze_file.flush()
    if debug_rows:
        debug_gaze_writer.writerows(debug_rows)
        debug_gaze_file.flush()


def quit_exp():
    try:
        flush_gaze()
        rec.stop()
    except Exception:
        pass
    for f in (trial_file, gaze_file, event_file, cal_file, debug_gaze_file):
        try:
            f.flush()
            f.close()
        except Exception:
            pass
    try:
        win.close()
    except Exception:
        pass
    core.quit()


def check_escape():
    keys = kb.getKeys(keyList=[QUIT_KEY], waitRelease=False, clear=True)
    if keys:
        quit_exp()


def show_message(text, wait_keys=(CONTINUE_KEY,)):
    message.text = text
    message.draw()
    win.flip()
    kb.clearEvents()
    while True:
        keys = kb.getKeys(keyList=list(wait_keys) + [QUIT_KEY], waitRelease=False)
        if any(k.name == QUIT_KEY for k in keys):
            quit_exp()
        for key in keys:
            if key.name in wait_keys:
                return key.name
        core.wait(0.005)



# ---------- integrated questionnaires (one item per screen) ----------
questionnaire_path_pre = None
questionnaire_path_post = None

SHAME_ITEMS = [
    'I want to sink into the floor and disappear.',
    'I feel small.',
    'I feel like I am a bad person.',
    'I feel humiliated, disgraced.',
    'I feel worthless, powerless.'
]
BELONG_ITEMS = ['I feel included.', 'I feel accepted.', 'I feel like I belong.']
CFQ_ITEMS = [
    'Do you find you forget why you went from one part of the house to the other?',
    "Do you find you forget whether you've turned off a light or a fire or locked the door?",
    'Do you fail to hear people speaking to you when you are doing something else?',
    'Do you find you forget appointments?',
    'Do you find you accidentally throw away the thing you want and keep what you meant to throw away?',
    'Do you find you forget what you came to the shops to buy?',
    'Do you forget to take your keys with you?',
    'Do you forget passwords you need to remember?',
    'Do you receive a delivery and forget what you ordered?',
    'Do you call people by the wrong name?',
    "In conversations, do you forget people's name even when you know them well?",
    'Do you get the dates of things mixed up?',
    'Do you find you go to put objects in the wrong place, like putting something from the cupboard in the fridge?',
    'Do you leave home and later realise you have forgotten something you intended to bring with you?',
    'Do you think something is missing, only to later see it where you had already looked?'
]
SHAME_LABELS = ['1  Not feeling this way at all', '2', '3  Feeling this way somewhat', '4', '5  Feeling this way very strongly']
INTENSITY_LABELS = ['1  Not at all', '2  Slightly', '3  Moderately', '4  Very', '5  Extremely']
CFQ_LABELS = ['1  Never', '2  Very Rarely', '3  Occasionally', '4  Quite Often', '5  Very Often']

q_title = visual.TextStim(win, text='', units='norm', pos=(0, 0.72), height=0.055,
                          color='white', wrapWidth=1.75, alignText='center')
q_item = visual.TextStim(win, text='', units='norm', pos=(0, 0.22), height=0.065,
                         color='white', wrapWidth=1.65, alignText='center')
q_help = visual.TextStim(win, text='', units='norm', pos=(0, -0.02), height=0.035,
                         color='lightgrey', wrapWidth=1.7, alignText='center')
q_options = visual.TextStim(win, text='', units='norm', pos=(0, -0.38), height=0.042,
                            color='white', wrapWidth=1.8, alignText='center')
q_footer = visual.TextStim(win, text='', units='norm', pos=(0, -0.82), height=0.032,
                           color='lightgrey', wrapWidth=1.8, alignText='center')


def questionnaire_intro(title, instructions):
    show_message(title + '\n\n' + instructions + '\n\nPress SPACE to begin.')


def save_questionnaire_rows(stage, rows):
    path = stem + '_questionnaire_' + stage + '.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['participant', 'condition', 'order', 'stage',
                                         'section', 'item_number', 'item', 'response',
                                         'response_label', 'system_time_us'])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


def run_likert_section(stage, section, items, labels, scale_name, instruction, rows, start_index=1):
    questionnaire_intro(section, instruction)
    local_rows = {}
    i = 0
    selected = None
    while i < len(items):
        item = items[i]
        item_number = start_index + i
        selected = local_rows.get(i, {}).get('selected')
        q_title.text = '{}  |  {} of {}'.format(section, i + 1, len(items))
        q_item.text = item
        q_help.text = instruction + (('\nSelected response: ' + str(selected)) if selected is not None else '')
        q_options.text = '\n\n'.join(labels)
        q_footer.text = 'Press 1–5 to select; ENTER to confirm; LEFT ARROW to go back; ESC to stop.'
        confirmed = False
        moved_back = False
        while not confirmed:
            q_title.draw(); q_item.draw(); q_help.draw(); q_options.draw(); q_footer.draw()
            win.flip()
            keys = kb.getKeys(keyList=['1','2','3','4','5','return','left',QUIT_KEY],
                              waitRelease=False, clear=True)
            for key in keys:
                if key.name == QUIT_KEY:
                    quit_exp()
                elif key.name in ['1','2','3','4','5']:
                    selected = int(key.name)
                elif key.name == 'left':
                    if i > 0:
                        i -= 1
                        selected = local_rows.get(i, {}).get('selected')
                    else:
                        selected = None
                    confirmed = True
                    moved_back = True
                    break
                elif key.name == 'return' and selected is not None:
                    response_value = selected - 1 if scale_name == 'cfq' else selected
                    local_rows[i] = {
                        'selected': selected,
                        'row': {
                            'participant': participant, 'condition': condition, 'order': order,
                            'stage': stage, 'section': section, 'item_number': item_number,
                            'item': item, 'response': response_value,
                            'response_label': labels[selected-1],
                            'system_time_us': tr.get_system_time_stamp()
                        }
                    }
                    confirmed = True
                    mark('questionnaire_response_' + stage + '_' + section.replace(' ', '_'))
                    checkpoint = rows + [local_rows[k]['row'] for k in sorted(local_rows)]
                    save_questionnaire_rows(stage, checkpoint)
                    break
            core.wait(0.002)
        if moved_back:
            continue
        i += 1
    rows.extend(local_rows[k]['row'] for k in sorted(local_rows))
    save_questionnaire_rows(stage, rows)
    return [local_rows[k]['row']['response'] for k in sorted(local_rows)]


def run_numeric_item(stage, section, item, rows):
    questionnaire_intro(section, item + '\n\nEnter a whole number from 0 to 100, then press ENTER.')
    value = ''
    while True:
        q_title.text = section
        q_item.text = item
        q_help.text = 'Your answer: ' + (value if value else '_')
        q_options.text = 'Use number keys. BACKSPACE deletes.'
        q_footer.text = 'ENTER to confirm; ESC to stop.'
        q_title.draw(); q_item.draw(); q_help.draw(); q_options.draw(); q_footer.draw(); win.flip()
        keys = kb.getKeys(waitRelease=False, clear=True)
        for key in keys:
            name = key.name
            if name == QUIT_KEY:
                quit_exp()
            elif name == 'backspace':
                value = value[:-1]
            elif name == 'return':
                if value.isdigit() and 0 <= int(value) <= 100:
                    rows.append({'participant': participant, 'condition': condition, 'order': order,
                                 'stage': stage, 'section': section, 'item_number': 1,
                                 'item': item, 'response': int(value), 'response_label': str(int(value)),
                                 'system_time_us': tr.get_system_time_stamp()})
                    mark('questionnaire_numeric_' + stage)
                    save_questionnaire_rows(stage, rows)
                    return
            elif name.isdigit() and len(value) < 3:
                value += name


def run_text_item(stage, section, item, rows):
    questionnaire_intro(section, item + '\n\nType your answer. Press ENTER when finished.')
    value = ''
    allowed_punct = {'space': ' ', 'comma': ',', 'period': '.', 'apostrophe': "'",
                     'minus': '-', 'slash': '/', 'semicolon': ';', 'colon': ':'}
    while True:
        q_title.text = section
        q_item.text = item
        q_help.text = value[-100:] if value else '(Type your response below)'
        q_options.text = 'BACKSPACE = delete    ENTER = submit'
        q_footer.text = 'Please answer in a few words.'
        q_title.draw(); q_item.draw(); q_help.draw(); q_options.draw(); q_footer.draw(); win.flip()
        keys = kb.getKeys(waitRelease=False, clear=True)
        for key in keys:
            name = key.name
            if name == QUIT_KEY:
                quit_exp()
            elif name == 'backspace':
                value = value[:-1]
            elif name == 'return':
                if value.strip():
                    rows.append({'participant': participant, 'condition': condition, 'order': order,
                                 'stage': stage, 'section': section, 'item_number': 1,
                                 'item': item, 'response': value.strip(), 'response_label': value.strip(),
                                 'system_time_us': tr.get_system_time_stamp()})
                    mark('questionnaire_text_' + stage)
                    save_questionnaire_rows(stage, rows)
                    return
            elif name in allowed_punct:
                value += allowed_punct[name]
            elif len(name) == 1 and name.isalnum():
                value += name


def run_pre_questionnaire():
    rows = []
    mark('pre_questionnaire_start')
    run_likert_section('pre', 'State shame', SHAME_ITEMS, SHAME_LABELS, 'shame',
                       'Rate how you feel right at this moment.', rows)
    run_likert_section('pre', 'Belongingness', BELONG_ITEMS, INTENSITY_LABELS, 'belonging',
                       'Please rate how you feel RIGHT NOW.', rows)
    run_likert_section('pre', 'Everyday mistakes (CFQ 2.0)', CFQ_ITEMS, CFQ_LABELS, 'cfq',
                       'Think about the past six months. Select the response that best reflects your experience.', rows)
    save_questionnaire_rows('pre', rows)
    mark('pre_questionnaire_end')


def run_post_questionnaire():
    rows = []
    mark('post_questionnaire_start')
    run_likert_section('post', 'State shame', SHAME_ITEMS, SHAME_LABELS, 'shame',
                       'Rate how you feel right at this moment.', rows)
    run_likert_section('post', 'Belongingness', BELONG_ITEMS, INTENSITY_LABELS, 'belonging',
                       'Please rate how you feel RIGHT NOW.', rows)
    run_likert_section('post', 'The game you just played',
                       ['I felt ignored.', 'I felt excluded.'], INTENSITY_LABELS, 'manipulation_check',
                       'Think about the ball-toss game you just played.', rows)
    run_numeric_item('post', 'The game you just played',
                     'Out of 100%, about what percentage of the throws did you receive?', rows)
    run_text_item('post', 'One last question',
                  'In a few words, what do you think this study was about?', rows)
    save_questionnaire_rows('post', rows)
    mark('post_questionnaire_end')


# ---------- integrated Cyberball on the existing PsychoPy window ----------
def run_cyberball():
    mark('cyberball_start')
    duration_s = 180.0
    flight_s = 0.7
    cpu_delay = (0.8, 1.8)
    excluded_receives = 2
    start_holder = 'p2'
    pos = {'you': (0.0, -0.28), 'p2': (-0.38, 0.12), 'p3': (0.38, 0.12)}
    labels = {'you': 'You', 'p2': 'Player 2', 'p3': 'Player 3'}
    colours = {'you': (0.2, 0.6, 1.0), 'p2': (1.0, 0.7, 0.2), 'p3': (0.5, 0.9, 0.5)}
    heads = {k: visual.Circle(win, radius=0.05, units='norm', fillColor=colours[k],
                              lineColor=None, pos=(v[0], v[1] + 0.07)) for k, v in pos.items()}
    bodies = {k: visual.Rect(win, width=0.13, height=0.09, units='norm', fillColor=colours[k],
                             lineColor=None, pos=(v[0], v[1] - 0.02)) for k, v in pos.items()}
    name_stims = {k: visual.TextStim(win, text=labels[k], height=0.035, units='norm', color='white',
                                     pos=(v[0], v[1] - 0.1)) for k, v in pos.items()}
    ball = visual.Circle(win, radius=0.022, units='norm', fillColor='white', lineColor=None)
    hint = visual.TextStim(win, text='', height=0.035, units='norm', color='white', pos=(0, -0.42))
    cb_msg = visual.TextStim(win, text='', height=0.045, units='norm', wrapWidth=1.55,
                             color='white', alignText='center')

    def hand(k):
        x, y = pos[k]
        return (x, y + 0.14)

    def draw_scene(ball_pos, hint_text=''):
        for k in pos:
            bodies[k].draw(); heads[k].draw(); name_stims[k].draw()
        ball.pos = ball_pos; ball.draw()
        if hint_text:
            hint.text = hint_text; hint.draw()

    def cyber_text(txt):
        cb_msg.text = txt; cb_msg.draw(); win.flip()
        kb.clearEvents()
        while True:
            keys = kb.getKeys(keyList=['space', QUIT_KEY], waitRelease=False, clear=True)
            if any(k.name == QUIT_KEY for k in keys): quit_exp()
            if any(k.name == 'space' for k in keys): return
            core.wait(0.005)

    def fly(src, dst):
        n = max(1, int(round(flight_s * fps)))
        x0, y0 = hand(src); x1, y1 = hand(dst)
        for i in range(1, n + 1):
            f = i / n
            draw_scene((x0 + (x1-x0)*f, y0 + (y1-y0)*f))
            win.flip(); check_escape()

    cyber_text('MENTAL VISUALISATION TASK\n\nYou will play a ball-toss game with two other players who are connected to the same game.\n\nPlease picture the people and the setting as vividly as you can.\n\nWhen you have the ball, press LEFT to throw to Player 2 or RIGHT to throw to Player 3.\n\nPress SPACE to start.')
    cb_path = stem + '_cyberball.csv'
    with open(cb_path, 'w', newline='', encoding='utf-8') as f:
        fields = ['participant', 'condition', 'throw', 'time_s', 'thrower', 'receiver',
                  'participant_receipts_so_far', 'system_time_us']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        holder = start_holder; receipts = 0; throw_n = 0
        game_clock = core.Clock(); kb.clearEvents()
        while game_clock.getTime() < duration_s:
            check_escape()
            if holder == 'you':
                receiver = None; kb.clearEvents()
                while receiver is None:
                    draw_scene(hand('you'), 'Press LEFT to throw to Player 2, RIGHT to throw to Player 3')
                    win.flip()
                    keys = kb.getKeys(keyList=['left', 'right', QUIT_KEY], waitRelease=False, clear=True)
                    for k in keys:
                        if k.name == QUIT_KEY: quit_exp()
                        if k.name == 'left': receiver = 'p2'
                        elif k.name == 'right': receiver = 'p3'
                    core.wait(0.002)
            else:
                delay = random.uniform(*cpu_delay); t0 = core.getTime()
                while core.getTime() - t0 < delay:
                    draw_scene(hand(holder)); win.flip(); check_escape()
                other = 'p3' if holder == 'p2' else 'p2'
                if condition == 'included':
                    receiver = random.choice(['you', other])
                else:
                    receiver = 'you' if receipts < excluded_receives else other
            throw_n += 1
            ts = tr.get_system_time_stamp()
            writer.writerow({'participant': participant, 'condition': condition, 'throw': throw_n,
                             'time_s': round(game_clock.getTime(), 3), 'thrower': holder,
                             'receiver': receiver, 'participant_receipts_so_far': receipts,
                             'system_time_us': ts})
            f.flush(); mark('cyberball_throw')
            fly(holder, receiver)
            if receiver == 'you': receipts += 1
            holder = receiver
        cyber_text('The game is over.\n\nPlease continue to the next part of the study.\n\nPress SPACE.')
    mark('cyberball_end')


def current_gaze_norm(row):
    if row is None:
        return None
    pts = []
    # Tobii Pro SDK Python validity: 1/True = valid; 0/False = invalid.
    if row[4] == 1 and row[2] is not None and row[3] is not None:
        try:
            x, y = float(row[2]), float(row[3])
            if math.isfinite(x) and math.isfinite(y):
                pts.append((x, y))
        except (TypeError, ValueError):
            pass
    if row[7] == 1 and row[5] is not None and row[6] is not None:
        try:
            x, y = float(row[5]), float(row[6])
            if math.isfinite(x) and math.isfinite(y):
                pts.append((x, y))
        except (TypeError, ValueError):
            pass
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts))


def median(values):
    values = sorted(values)
    n = len(values)
    if not n:
        return None
    return values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])

# ---------- calibration / validation ----------
CAL_POINTS = [(0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
VAL_POINTS = [(0.5, 0.5), (0.15, 0.5), (0.85, 0.5), (0.5, 0.15), (0.5, 0.85)]


def calibration():
    show_message('CALIBRATION\n\nLook directly at each dot as it appears. Keep your head still.\n\nPress SPACE to begin.')
    calib = tr.ScreenBasedCalibration(rec.tracker)
    outer = visual.Circle(win, radius=0.45, fillColor='white', lineColor='white')
    inner = visual.Circle(win, radius=0.12, fillColor='black', lineColor='black')
    calib.enter_calibration_mode()
    try:
        for xn, yn in CAL_POINTS:
            outer.pos = inner.pos = norm_to_deg(xn, yn)
            t0 = core.getTime()
            while core.getTime() - t0 < 0.9:
                outer.draw(); inner.draw(); win.flip(); check_escape()
            status = calib.collect_data(xn, yn)
            if status != tr.CALIBRATION_STATUS_SUCCESS:
                print('Calibration point collection status:', status)
        result = calib.compute_and_apply()
        print('Calibration result:', result.status)
    finally:
        calib.leave_calibration_mode()
    rec.clear()


def validation(stage, duration=0.8):
    """Brief 5-point validation; experimenter accepts/rejects after viewing errors."""
    show_message('TRACKING CHECK\n\nLook at each dot as accurately as possible.\n\nPress SPACE to begin.')
    dot = visual.Circle(win, radius=0.25, fillColor='white', lineColor='white')
    errors = []
    for i, (xn, yn) in enumerate(VAL_POINTS, start=1):
        dot.pos = norm_to_deg(xn, yn)
        dot.draw(); win.flip()
        t0 = tr.get_system_time_stamp()
        t_end = core.getTime() + duration
        while core.getTime() < t_end:
            dot.draw(); win.flip(); check_escape()
        t1 = tr.get_system_time_stamp()
        rows = [r for r in rec.snapshot() if r[0] is not None and t0 <= r[0] <= t1]
        pts = [current_gaze_norm(r) for r in rows]
        pts = [p for p in pts if p is not None
               and math.isfinite(p[0]) and math.isfinite(p[1])]
        gx = median([p[0] for p in pts]) if pts else None
        gy = median([p[1] for p in pts]) if pts else None
        if gx is not None:
            tx_deg, ty_deg = norm_to_deg(xn, yn)
            gx_deg, gy_deg = norm_to_deg(gx, gy)
            err = math.hypot(gx_deg - tx_deg, gy_deg - ty_deg)
        else:
            err = None
        errors.append(err)
        cal_writer.writerow([stage, i, xn, yn, gx, gy, err, len(pts)])
    cal_file.flush()
    # Flush raw SDK values before presenting the validation result.
    flush_gaze()
    usable = [e for e in errors if e is not None and math.isfinite(e)]
    if usable:
        summary = ('Mean error: {:.2f} deg; worst: {:.2f} deg; usable points: {}/5'
                   .format(sum(usable) / len(usable), max(usable), len(usable)))
    else:
        summary = 'No finite valid gaze measurements. Check tracking and validation sampling.'
    point_summary = 'Per point: ' + ', '.join(
        '-' if e is None else ('{:.2f}'.format(e) if math.isfinite(e) else 'invalid')
        for e in errors)
    choice = show_message('VALIDATION RESULTS\n\n' + summary +
                          '\n' + point_summary +
                          '\n\nSPACE = accept and continue\nR = recalibrate',
                          wait_keys=(CONTINUE_KEY, 'r'))
    # Apply a pre-specified lab acceptance criterion; this display is not a guarantee.
    rec.clear()
    return choice, errors

# ---------- task design ----------


def present_for_duration(stimulus, duration_s, onset_event, offset_event):
    """Present a stimulus for an integer number of refresh frames."""
    n_frames = max(1, int(round(duration_s * fps)))
    onset_us = None
    for frame in range(n_frames):
        stimulus.draw()
        if frame == 0:
            win.flip()
            onset_us = mark(onset_event)
        else:
            win.flip()
        check_escape()
    win.flip()
    offset_us = mark(offset_event)
    actual_s = n_frames / fps
    return onset_us, offset_us, actual_s

def make_trials(task, n_trials, practice=False):
    """Balanced cue side and bracket direction; target side follows task rule."""
    n_left = n_trials // 2
    cue_sides = ['left'] * n_left + ['right'] * (n_trials - n_left)
    random.shuffle(cue_sides)
    brackets = ['{', '}'] * (n_trials // 2)
    if n_trials % 2:
        brackets.append(random.choice(['{', '}']))
    random.shuffle(brackets)
    trials = []
    for cue_side, bracket in zip(cue_sides, brackets):
        if task == 'pro':
            target_side = cue_side
        else:
            target_side = 'right' if cue_side == 'left' else 'left'
        trials.append({'cue_side': cue_side, 'target_side': target_side,
                       'central_bracket': bracket,
                       'peripheral_bracket': random.choice(['{', '}'])})
    random.shuffle(trials)
    return trials


def response_for_bracket(bracket):
    return LEFT_RESPONSE_KEY if bracket == '{' else RIGHT_RESPONSE_KEY


def collect_response(timeout_s, onset_event):
    """Collect left/right arrow response; RT is relative to screen flip."""
    kb.clearEvents()
    clock = core.Clock()
    response = None
    rt = None
    clock.reset()
    mark(onset_event)
    while clock.getTime() < timeout_s:
        keys = kb.getKeys(keyList=[LEFT_RESPONSE_KEY, RIGHT_RESPONSE_KEY, QUIT_KEY],
                          waitRelease=False)
        for key in keys:
            if key.name == QUIT_KEY:
                quit_exp()
            if response is None and key.name in (LEFT_RESPONSE_KEY, RIGHT_RESPONSE_KEY):
                response = key.name
                rt = clock.getTime()
                return response, rt
        core.wait(0.001)
    return response, rt


def run_one_trial(task, stage, block_num, trial_num, spec, practice=False):
    CUR.update(stage=stage, block_number=block_num, task=task,
               trial_number=trial_num, practice=int(practice))
    mark('trial_start')

    # 1. Central fixation with a random foreperiod (adaptation; exact original
    # random-delay distribution was not specified in the paper).
    foreperiod = random.uniform(FIXATION_MIN, FIXATION_MAX)
    fix.draw(); win.flip(); mark('fixation_onset')
    core.wait(foreperiod)

    # 2. Central bracket remains until response (original sequence).
    central = spec['central_bracket']
    central_bracket_stim.text = central
    central_bracket_stim.pos = (0, 0)
    central_bracket_stim.draw(); win.flip()
    central_onset = mark('central_target_onset')
    central_resp, central_rt = collect_response(CENTRAL_RESPONSE_TIMEOUT,
                                                'central_response_window')
    mark('central_target_offset')
    central_correct = int(central_resp == response_for_bracket(central)) if central_resp else 0

    # 3. Peripheral cue, 300 ms.
    cue_x = -ECCENTRICITY_DEG if spec['cue_side'] == 'left' else ECCENTRICITY_DEG
    cue_stim.pos = (cue_x, 0)
    cue_stim.pos = (cue_x, 0)
    cue_onset, cue_offset, actual_cue_dur = present_for_duration(
        cue_stim, CUE_DUR, 'cue_onset', 'cue_offset')

    # 4. Peripheral bracket target, 100 ms, on same side (pro) or opposite (anti).
    target_x = -ECCENTRICITY_DEG if spec['target_side'] == 'left' else ECCENTRICITY_DEG
    peripheral = spec['peripheral_bracket']
    peripheral_bracket_stim.text = peripheral
    peripheral_bracket_stim.pos = (target_x, 0)
    peripheral_onset, peripheral_offset, actual_target_dur = present_for_duration(
        peripheral_bracket_stim, PERIPHERAL_TARGET_DUR,
        'peripheral_target_onset', 'peripheral_target_offset')

    # 5. Second forced-choice response after the brief target has disappeared.
    peripheral_resp, peripheral_rt = collect_response(
        PERIPHERAL_RESPONSE_TIMEOUT, 'peripheral_response_window')
    peripheral_correct = int(peripheral_resp == response_for_bracket(peripheral)) if peripheral_resp else 0
    mark('trial_end')

    row = {
        'participant': participant, 'condition': condition, 'order': order,
        'stage': stage, 'block_number': block_num, 'task': task,
        'trial_number': trial_num, 'practice': int(practice),
        'cue_side': spec['cue_side'], 'target_side': spec['target_side'],
        'central_bracket': central, 'central_response': central_resp or '',
        'central_correct': central_correct,
        'central_rt_s': '' if central_rt is None else round(central_rt, 4),
        'peripheral_bracket': peripheral,
        'peripheral_response': peripheral_resp or '',
        'peripheral_correct': peripheral_correct,
        'peripheral_rt_s': '' if peripheral_rt is None else round(peripheral_rt, 4),
        'central_omission': int(central_resp is None),
        'peripheral_omission': int(peripheral_resp is None),
        'fixation_foreperiod_s': round(foreperiod, 4),
        'cue_onset_system_us': cue_onset, 'cue_offset_system_us': cue_offset,
        'peripheral_target_onset_system_us': peripheral_onset,
        'peripheral_target_offset_system_us': peripheral_offset
    }
    if not practice:
        trial_writer.writerow(row)
        trial_file.flush()
    flush_gaze()


def run_block(task, stage, block_num, n_trials, practice=False):
    trials = make_trials(task, n_trials, practice)
    for i, spec in enumerate(trials, start=1):
        run_one_trial(task, stage, block_num, i, spec, practice=practice)


def block_instructions(task, practice=False):
    if task == 'pro':
        task_text = ('PROSACCADE: When the square appears, move your eyes towards it. '
                     'The bracket will appear at that same side.')
    else:
        task_text = ('ANTISACCADE: When the square appears, inhibit looking towards it. '
                     'Move your eyes to the opposite side, where the bracket will appear.')
    practice_text = 'You will complete a short practice.' if practice else 'There will be no correctness feedback.'
    show_message(task_text + '\n\nFor BOTH brackets, identify its direction:\n'
                 '{ = LEFT ARROW;   } = RIGHT ARROW.\n\n' + practice_text +
                 '\n\nKeep your head still. Press SPACE when ready.')

# ---------- run session ----------
try:
    rec.start()
    show_message('WELCOME\n\nPlease sit comfortably, keep your head as still as possible, and follow the instructions.\n\nPress SPACE when ready.')

    half_width_deg = (SCREEN_WIDTH_CM / 2.0) / CM_PER_DEG
    if ECCENTRICITY_DEG >= half_width_deg:
        raise RuntimeError('ECCENTRICITY_DEG is outside the measured screen. Reduce it or correct screen geometry.')
    stimulus_cm = VIEW_DIST_CM * math.tan(math.radians(ECCENTRICITY_DEG))
    stimulus_px = stimulus_cm * SCREEN_RES[0] / SCREEN_WIDTH_CM
    edge_margin_cm = SCREEN_WIDTH_CM / 2.0 - stimulus_cm
    print('Monitor profile: {}'.format(MONITOR_NAME))
    print('Configured resolution: {} x {}'.format(*SCREEN_RES))
    print('Window-reported resolution: {}'.format(tuple(win.size)))
    print('Screen width: {:.1f} cm; viewing distance: {:.1f} cm'.format(SCREEN_WIDTH_CM, VIEW_DIST_CM))
    print('Stimulus position: {:.2f} cm / {:.1f} px from centre; edge margin: {:.2f} cm'.format(
        stimulus_cm, stimulus_px, edge_margin_cm))
    if tuple(win.size) != SCREEN_RES:
        raise RuntimeError('Actual window resolution does not match configured 1366 x 768 display.')

    # Baseline questionnaires occur before calibration and the baseline eye task.
    run_pre_questionnaire()

    calibration()
    choice, pre_validation_errors = validation('pre')
    if choice == 'r':
        calibration()
        validation('pre_repeat')

    practice_order = ['pro', 'anti'] if order == 'pro_anti' else ['anti', 'pro']
    for b, task in enumerate(practice_order, start=1):
        block_instructions(task, practice=True)
        run_block(task, 'practice', b, N_PRACTICE_PER_TASK, practice=True)

    pre_blocks = ['pro', 'anti'] if order == 'pro_anti' else ['anti', 'pro']
    show_message('BASELINE EYE-TRACKING\n\nThe two baseline blocks are about to begin.\n\nPress SPACE.')
    mark('pre_eye_task_start')
    for b, task in enumerate(pre_blocks, start=1):
        block_instructions(task, practice=False)
        run_block(task, 'pre', b, N_TEST_TRIALS_PER_BLOCK, practice=False)
        if b == 1:
            show_message('Block complete. Take a brief rest.\n\nPress SPACE to continue.')
    mark('pre_eye_task_end')

    # Cyberball condition and task order were allocated together using balanced blocks.
    run_cyberball()
    run_post_questionnaire()

    # Revalidate after the manipulation and questionnaire before post-task recording.
    if CHECK_AFTER_CYBERBALL:
        choice, post_validation_errors = validation('post')
        if choice == 'r':
            calibration()
            validation('post_repeat')

    show_message('POST-MANIPULATION EYE-TRACKING\n\nThe final two blocks will now begin in the same order as baseline.\n\nPress SPACE.')
    mark('post_eye_task_start')
    for b, task in enumerate(pre_blocks, start=3):
        block_instructions(task, practice=False)
        run_block(task, 'post', b, N_TEST_TRIALS_PER_BLOCK, practice=False)
        if b == 3:
            show_message('Block complete. Take a brief rest.\n\nPress SPACE to continue.')
    mark('post_eye_task_end')

    show_message('TASK COMPLETE\n\nPlease inform the experimenter. The experimenter will now debrief you.\n\nPress SPACE to save and close.')
    mark('session_complete')
    quit_exp()

except Exception as exc:
    print('ERROR:', repr(exc))
    try:
        show_message('The task encountered an error. Please inform the experimenter.\n\n'
                     'Press SPACE to close and save the data.')
    except Exception:
        pass
    quit_exp()
