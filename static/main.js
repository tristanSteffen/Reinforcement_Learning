// static/main.js

const boardElement = document.getElementById("board");
const statusElement = document.getElementById("status");

// We'll store the board as an array of 0 (empty), 1 (X, AI), 2 (O, human)
let boardState = [0, 0, 0, 0, 0, 0, 0, 0, 0];
let gameOver = false;

// Render the board
function renderBoard() {
  boardElement.innerHTML = "";
  boardState.forEach((cell, idx) => {
    const cellDiv = document.createElement("div");
    cellDiv.classList.add("cell");
    cellDiv.textContent = cell === 1 ? "X" : cell === 2 ? "O" : "";
    cellDiv.addEventListener("click", () => handleClick(idx));
    boardElement.appendChild(cellDiv);
  });
}

// Check if there's a winner or draw
function checkGameOver() {
  const wins = [
    [0,1,2], [3,4,5], [6,7,8],
    [0,3,6], [1,4,7], [2,5,8],
    [0,4,8], [2,4,6]
  ];
  for (let [a,b,c] of wins) {
    if (boardState[a] !== 0 &&
        boardState[a] === boardState[b] &&
        boardState[b] === boardState[c]) {
      return boardState[a]; // 1 or 2
    }
  }
  // Check draw
  if (!boardState.includes(0)) {
    return 0; // draw
  }
  return null; // not finished
}

function handleClick(idx) {
  if (gameOver) return;
  if (boardState[idx] !== 0) return;

  // Human is O (2)
  boardState[idx] = 2;
  renderBoard();

  let winner = checkGameOver();
  if (winner !== null) {
    endGame(winner);
    return;
  }

  // Now request AI move from server
  fetch("/get_ai_move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ board: boardState })
  })
    .then((res) => res.json())
    .then((data) => {
      const aiMove = data.action;
      if (aiMove === -1) {
        // No move or error
        let winnerCheck = checkGameOver();
        endGame(winnerCheck);
      } else {
        boardState[aiMove] = 1; // AI is X
        renderBoard();
        let winnerCheck = checkGameOver();
        if (winnerCheck !== null) {
          endGame(winnerCheck);
        }
      }
    })
    .catch((err) => console.error(err));
}

function endGame(winner) {
  gameOver = true;
  if (winner === 1) {
    statusElement.textContent = "AI (X) wins!";
  } else if (winner === 2) {
    statusElement.textContent = "You (O) win!";
  } else {
    statusElement.textContent = "It's a draw!";
  }
}

// Initial render
renderBoard();
