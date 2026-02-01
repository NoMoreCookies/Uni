import { use, useEffect, useState } from 'react';
import './App.css';

/**
 * 
 * @param {Array} squares 
 * @returns {string} winner sign (if draw or no winner returns null
 */

function calculateWinner(squares) {

  const lines = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6]
  ];

  for (let i = 0; i < lines.length; i++) {

    const [a, b, c] = lines[i];

    if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) {

      return squares[a];

    }

  }
  return null;
}


/**
 * Post request for AlphaBeta alg 
 * 
 * @param {Array} board current game state
 * @param {string} aiSymbol 
 * @returns {Array}  new board state after ai move
 */
async function getBestMove(board, aiSymbol){

  const res = await fetch("http://localhost:8000/api/best-move", { 
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ board, ai: aiSymbol }),

  });

  // error resistance
  if(!res.ok) throw new Error(await res.text());

  // await bo musimy poczekać aż pobierze się cały json
  const data = await res.json();

  return data;
}



/**
 * 
 * @param {*} param0 
 * @returns {button} 3d button object (clicked-or-not)
 */
function Square({ value, onSquareClick, disabled}) {

  // czy jest już kliknięty
  const isDown = value !== null;

  return (
    <button 
      className={`btn-3d ${isDown ? "down" : ""}`} 
      onClick = {onSquareClick} 
      disabled = {disabled || isDown}>
      {value}
    </button>
  );

}


/**
 * 
 * @param {*} param0 
 * @returns 
 */
function Board({xIsNext, squares, onPlay, aiEnabled, aiSymbol }) {

  const [aiThinking, setAiThinking] = useState(false);


  // prevents from makeing move when it is not our turn
  function handleClick(i) {
    if (aiThinking) return;

    if (squares[i] || calculateWinner(squares)) {
      return;
    }


    //creating copy of array
    const nextSquares = squares.slice();

    // checks who's turn is now
    nextSquares[i] = xIsNext ? "X" : "O"

    // update of game state
    onPlay(nextSquares)
  }

  useEffect(()=>{

    // block scoped variable let 
    let cancelled = false;

    const winner = calculateWinner(squares);

    const boardFull = squares.every(s => s !== null);

    if(!aiEnabled || winner || boardFull) return;

    // whos turn is that
    const turn = xIsNext? "X" : "O"

    // if this is not ai's turn end
    if(turn !== aiSymbol) return; 

    (async () => {

      try{

        setAiThinking(true);

        const data = await getBestMove(squares, aiSymbol);

        if (cancelled) return;

        if(data.move == null) return;

        onPlay(data.board_after); 

      } finally{

        if(!cancelled) setAiThinking(false);

      }
    })();
    return () => {cancelled = true;} ;
}, 

[aiEnabled, squares, xIsNext, onPlay, aiSymbol]);

  return (
    <div className='app'>
        <div className="board-row">
          <Square value={squares[0]} onSquareClick={() => handleClick(0)} />
          <Square value={squares[1]} onSquareClick={() => handleClick(1)} />
          <Square value={squares[2]} onSquareClick={() => handleClick(2)} />
        </div>
        <div className="board-row">
          <Square value={squares[3]} onSquareClick={() => handleClick(3)} />
          <Square value={squares[4]} onSquareClick={() => handleClick(4)} />
          <Square value={squares[5]} onSquareClick={() => handleClick(5)} />
        </div>
        <div className="board-row">
          <Square value={squares[6]} onSquareClick={() => handleClick(6)} />
          <Square value={squares[7]} onSquareClick={() => handleClick(7)} />
          <Square value={squares[8]} onSquareClick={() => handleClick(8)} />
        </div>
      </div>
  );
}


export default function App(){

  const [history, setHistory] = useState([Array(9).fill(null)]);

  const [currentMove, setCurrentMove] = useState(0);

  const [aiEnabled,setAiEnabled] = useState(true);

  const aiSymbol = "O";

  const xIsNext = currentMove % 2 === 0 ;
  const squares = history[currentMove];

  
  function handlePlay(nextSquares){

    const nextHistory= [...history.slice(0, currentMove + 1), nextSquares];

    setHistory(nextHistory);

    setCurrentMove(nextHistory.length - 1);

    }

    const winner = calculateWinner(squares);

    const status = winner

    ? `Winner: ${winner}`
    : `Next player: ${xIsNext ? "X" : "O"}`;

    return (
      <div className="game-container">

        <label className="status">{status}</label>

        <label className="toggle">
          <input
            type="checkbox"
            checked={aiEnabled}
            onChange={(e) => setAiEnabled(e.target.checked)}
          />
          <span className="slider" />
          <span className="label-text">Play vs AI</span>
        </label>

        <Board
          xIsNext={xIsNext}
          squares={squares}
          onPlay={handlePlay}
          aiEnabled={aiEnabled}
          aiSymbol={aiSymbol}
        />
      </div>

    );
}







