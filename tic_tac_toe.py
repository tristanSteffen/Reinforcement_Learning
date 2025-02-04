import numpy as np

class TicTacToe:
    """
    A Tic-Tac-Toe environment where:
      -  1 = X
      - -1 = O
      -  0 = empty
    """
    def __init__(self):
        self.board = np.zeros(9, dtype=int)  # Flatten for convenience (size=9).
        self.done = False
        self.winner = None
        self.current_player = 1  # X starts
        self.reset()

    def reset(self):
        self.board[:] = 0
        self.done = False
        self.winner = None
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        # Return a copy of the board as the current state
        return self.board.copy()

    def available_actions(self):
        """
        Return the list of valid actions (0..8) that are currently empty.
        """
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, action):
        """
        Execute 'action' (an int 0..8), place current_player's mark,
        then check if game has ended.
        Returns (next_state, reward, done).
        """
        if self.board[action] != 0 or self.done:
            # Illegal move or game already finished
            return self.get_state(), 0, True

        # Place the mark
        self.board[action] = self.current_player

        # Check winner
        self.check_winner()
        if self.done:
            if self.winner == 1:
                return self.get_state(), 5, True
            elif self.winner == -1:
                return self.get_state(), -2, True
            else:
                # Draw
                return self.get_state(), 0.5, True
        else:
            # Switch player: 1 -> -1, -1 -> 1
            self.current_player = -1 if self.current_player == 1 else 1
            return self.get_state(), -0.1, False

    def check_winner(self):
        """
        Check if there's a winner or a draw.
        Sets self.done=True and self.winner accordingly:
           self.winner = 1  (X wins)
           self.winner = -1 (O wins)
           self.winner = 0  (draw)
        or None if the game not finished.
        """
        # Reshape for checking lines
        b = self.board.reshape(3, 3)

        # Rows, columns, diagonals
        for i in range(3):
            row_sum = sum(b[i, :])
            col_sum = sum(b[:, i])
            if row_sum == 3 or col_sum == 3:
                self.done = True
                self.winner = 1
                return
            elif row_sum == -3 or col_sum == -3:
                self.done = True
                self.winner = -1
                return

        diag1 = b[0, 0] + b[1, 1] + b[2, 2]
        diag2 = b[0, 2] + b[1, 1] + b[2, 0]
        if diag1 == 3 or diag2 == 3:
            self.done = True
            self.winner = 1
            return
        elif diag1 == -3 or diag2 == -3:
            self.done = True
            self.winner = -1
            return

        # Check draw
        if 0 not in self.board:
            self.done = True
            self.winner = 0
