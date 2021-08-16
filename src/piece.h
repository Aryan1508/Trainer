#pragma once
#include <iostream>

enum Color : uint8_t
{
    White,
    Black
};

enum PieceType : uint8_t
{
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King
};

enum Piece : uint8_t
{
    wPawn,
    wKnight,
    wBishop,
    wRook,
    wQueen,
    wKing,
    bPawn,
    bKnight,
    bBishop,
    bRook,
    bQueen,
    bKing,
    Empty
};

inline Piece make_piece(PieceType type, Color color)
{
    return static_cast<Piece>(static_cast<uint8_t>(type) + (static_cast<uint8_t>(color) * 6));
}