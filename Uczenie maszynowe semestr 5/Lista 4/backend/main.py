from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Literal, Tuple, Any
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials = True,
    allow_methods =["*"],
    allow_headers=["*"],
)

Player = Literal["X","O"]

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]


# to co dostaje z fronta
class BestMoveRequest(BaseModel):
    board: List[Any]
    ai: Player


# to co odsyła backend (to jest plansza po ruchu AI)
class BestMoveResponse(BaseModel):
    move: Optional[int]
    board_after : List[Optional[Player]]
    winner: Optional[Player]
    draw: bool
    score: int


# tworzenie tablicy na podstawie frontendu
def normalize_board( raw: List[Any] ) -> List[Optional[Player]]:
    out: List[Optional[Player]] = []
    for v in raw:
        if v is None or v == "" or v == 0:
            out.append(None)
        elif v in ("X","x",1,"1"):
            out.append("X")
        elif v in ("O","o",2,"2"):
            out.append("O")
    return out


# sprawdzanie czy mamy wygranego
def winner_of(board: List[Optional[Player]]) -> bool:
    for a,b,c in WIN_LINES:
        if board[a] is not None and board[a] == board[b] == board[c]:
            return board[a]
    return None


# czy mamy remis
def is_draw(board: List[Optional[Player]]) -> bool:
    return winner_of(board) is None and all(cell is not None for cell in board)


# jakie mamy dostępne ruchy (które pola są None)
def available_moves(board: List[Optional[Player]]) -> List[int]:
    return [i for i, v in enumerate(board) if v is None]


# kto jest poprzednim graczem
def other(p: Player) -> Player:
    return "O" if p == "X" else "X"


# efekt końcowy
def terminal_score(board: List[Optional[Player]], ai: Player) -> Optional[int]:
    w = winner_of(board)
    if w == ai:
        return 1
    if w == other(ai):
        return -1
    if is_draw(board):
        return 0
    return None


# algorytm AlphaBeta
def AlphaBeta(
    board : List[Optional[Player]],# position
    turn : Player, # czyja jest tura
    ai: Player, # czy gramy kółkiem czy krzyzykiem
    alpha: int, # alpha
    beta: int, # beta
) -> int:
    tscore = terminal_score(board, ai)
    if tscore is not None:
        return tscore
    
    moves = available_moves(board)

    if turn == ai:

        value  = -10

        for m in moves:

            board[m] = turn

            value = max(value, AlphaBeta(board, other(turn), ai, alpha, beta))

            board[m] = None

            alpha = max(alpha, value)

            if alpha>=beta:
                break

        return value
    else:

        value = 10
        
        for m in moves:

            board[m] = turn

            value = min(value, AlphaBeta(board, other(turn), ai, alpha, beta))

            board[m] = None

            beta = min(beta, value)

            if alpha >= beta:
                break

        return value


def best_move(board: List[Optional[Player]], ai: Player) -> Tuple[Optional[int], int]:

    
    tscore = terminal_score(board, ai)

    if tscore is not None:
        return None, tscore
    
    best_m = None
    best_s = -10

    for m in available_moves(board):
        board[m] = ai
        s = AlphaBeta(board, other(ai), ai, alpha = -10, beta = 10)
        board[m] = None

        if s > best_s:
            best_s = s
            best_m = m 
    return best_m, best_s



@app.get("/api/health")
def health():
        return {"ok": True}

    
@app.post("/api/best-move", response_model = BestMoveResponse)
def api_best_move(req: BestMoveRequest):
        
        board = normalize_board(req.board)

        ai = req.ai

        move, score = best_move(board, ai)

        print("Best move debug:", board, "->", move, score)

        board_after = board[:]

        if move is not None:

            board_after[move] = ai

        w = winner_of(board_after)
        d = is_draw(board_after)

        time.sleep(2)

        return BestMoveResponse(
            move = move,
            board_after = board_after,
            winner = w,
            draw = d,
            score = score
        )