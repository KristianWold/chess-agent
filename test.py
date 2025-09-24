def roundtrip_check(agent, board):
    ok = True
    for m in board.legal_moves:
        a = agent.board_logic.move_to_action(m)
        m2 = agent.board_logic.action_to_move(a)
        # Normalize for chess.Move equality (python-chess compares exactly)
        if m != m2:
            print("Mismatch:", m, "->", a, "->", m2)
            ok = False
    return ok

assert roundtrip_check(agent, environment.get_board())
