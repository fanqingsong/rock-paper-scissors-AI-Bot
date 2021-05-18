from keras.models import load_model
import cv2
import numpy as np
from random import choice

########################### Reinforcement Learning S ##################################

class Bot(object):
    # our states can be either "ROCK, PAPER or SCISSORS"
    state_space = 3

    # three actions by our player
    action_space = 3

    q_table = np.random.uniform(low=-2, high=5, size=(3, 3))
    total_reward, reward = 0, 0
    avg_rewards_list = []
    avg_reward = 0
    result = 'DRAW'
    tags = ["R", "P", "S"]
    # looses to map
    loses_to = {
        "0": 1,  # rock loses to paper
        "1": 2,  # paper loses to scissor
        "2": 0  # scissor loses to rock
    }

    def __init__(self, alpha=0.5, gamma=0.2, epsilon=0.8, min_eps=0, episodes=1000, verbose=False):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.episodes = episodes
        # Calculate episodic reduction in epsilon
        self.reduction = (epsilon - min_eps) / episodes

        self.verbose = verbose

    # either explore or exploit, any which ways return the next action
    def bot_move(self, player_move):
        action = 0
        # Determine next action - epsilon greedy strategy
        if np.random.random() < 1 - self.epsilon:
            if self.verbose:
                print("Exploiting....")

            action = np.argmax(self.q_table[player_move])
        else:
            if self.verbose:
                print("Exploring.....")

            action = np.random.randint(0, self.action_space)

        # Decay epsilon
        if self.epsilon > self.min_eps:
            self.epsilon -= self.reduction

        if self.verbose:
            print("choose ", self.tags[action])

        return action

    def get_reward(self, player, bot):
        reward = 0

        if self.get_result(player, bot) == 'WIN':
            reward = 5
        elif self.get_result(player, bot) == 'LOSE':
            reward = -2
        else:
            # Draw case
            reward = 4

        return reward

    # update q_table
    def update_experience(self, state, action, reward, player_next_move):
        reward_next_move = np.max(self.q_table[player_next_move])
        delta = self.alpha * (reward + self.gamma * reward_next_move - self.q_table[state, action])
        self.q_table[state, action] += delta

    def print_stats(self, player, bot, reward):
        if self.verbose:
            print("Player move : {0}, bot: {1}, reward: {2}, result: {3}, total_reward: {4}".format(self.tags[player],
                                                                                                    self.tags[bot], reward,
                                                                                                    self.result,
                                                                                                    self.total_reward))
            print(self.q_table)

    # returns either a WIN, LOSE or a DRAW to indicate the same.
    def get_result(self, player_move, bot_move):
        if bot_move == player_move:
            self.result = 'DRAW'
        elif self.loses_to[str(bot_move)] == player_move:
            self.result = 'LOSE'
        else:
            self.result = 'WIN'

        return self.result

    def get_avg_rewards(self):
        return self.avg_rewards_list

    def learn(self, player_move, bot_move, player_next_move):
        # add reward
        reward = self.get_reward(player_move, bot_move)

        self.total_reward += reward
        self.avg_rewards_list.append(reward)

        # update experience
        self.update_experience(player_move, bot_move, reward, player_next_move)
        self.print_stats(player_move, bot_move, reward)


episodes = 50

bot_player = Bot(verbose=False, episodes=episodes)

opponent_history = []
me_history = []


def computer_player(opponent_prev_play, me_prev_play, verbose=False):
    # print("call player")
    # print(prev_play)
    # print(len(opponent_history))

    play_list = ["R", "P", "S"]
    win_dict = {"R": "P", "P": "S", "S": "R"}

    # suppose opponent's play is R, before real first round
    opponent_prev_play_index = 0

    if opponent_prev_play in play_list:
        if len(opponent_history) > 0:
            opponent_prev_prev_play = opponent_history[-1]
            opponent_prev_prev_play_index = play_list.index(opponent_prev_prev_play)

        opponent_history.append(opponent_prev_play)
        opponent_prev_play_index = play_list.index(opponent_prev_play)

    if me_prev_play in play_list:
        if len(me_history) > 1:
            me_prev_prev_play = me_history[-1]
            me_prev_prev_play_index = play_list.index(me_prev_prev_play)

        me_history.append(me_prev_play)

    if len(opponent_history) >= 3:
        state = opponent_prev_prev_play_index
        next_state = opponent_prev_play_index
        action = me_prev_prev_play_index
        bot_player.learn(state, action, next_state)

    me_play_index = bot_player.bot_move(opponent_prev_play_index)

    if verbose:
        print(f"opponent most possible next play is {me_play_index}")

    me_play = play_list[me_play_index]

    return me_play

########################### Reinforcement Learning E ##################################

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"


model = load_model("rock-paper-scissors-model.h5")

cap = cv2.VideoCapture(0)

# Set properties. Each returns === True on success (i.e. correct resolution)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1260)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

prev_move = None

prev_computer_move_name = ""
prev_user_move_name = ""

move_mapper_Bot2APP = {'R': 'rock',  'P': 'paper', 'S': 'scissors'}
move_mapper_APP2Bot = {'rock': 'R',  'paper': 'P', 'scissors': 'S'}

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user to play
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)

    # rectangle for computer to play
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)

    # extract the region of image within the user rectangle
    roi = frame[100:500, 100:500]

    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    # predict the winner (human vs computer)
    if prev_move != user_move_name:
        if user_move_name != "none":
            # computer_move_name = choice(['rock', 'paper', 'scissors'])

            if prev_user_move_name in move_mapper_APP2Bot:
                prev_user_move_name = move_mapper_APP2Bot[prev_user_move_name]

            if prev_computer_move_name in move_mapper_APP2Bot:
                prev_computer_move_name = move_mapper_APP2Bot[prev_computer_move_name]

            computer_move_name = computer_player(prev_user_move_name, prev_computer_move_name)

            computer_move_name = move_mapper_Bot2APP[computer_move_name]

            winner = calculate_winner(user_move_name, computer_move_name)

            prev_user_move_name = user_move_name
            prev_computer_move_name = computer_move_name
        else:
            computer_move_name = "none"
            winner = "Waiting..."
    prev_move = user_move_name

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    if computer_move_name != "none":
        icon = cv2.imread(
            "images/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (400, 400))
        print(icon.shape)
        print(frame.shape)
        frame[100:500, 800:1200] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
