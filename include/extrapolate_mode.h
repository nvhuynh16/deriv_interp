#pragma once

#ifndef EXTRAPOLATE_MODE_DEFINED
#define EXTRAPOLATE_MODE_DEFINED

/**
 * Extrapolation mode for piecewise polynomial evaluation
 * Shared by BPoly, CPoly, LegPoly, BsPoly, and LagPoly
 */
enum class ExtrapolateMode {
    Extrapolate,    // Use polynomial from nearest segment (default)
    NoExtrapolate,  // Return NaN for out-of-bounds
    Periodic        // Wrap around periodically
};

#endif
