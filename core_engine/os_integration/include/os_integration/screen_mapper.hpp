#pragma once

#include "os_integration/types.hpp"

namespace osi {

// The part of the camera's normalized space that should cover the whole
// target screen. A hand region smaller than the frame means small hand
// movements reach every screen edge (less "gorilla arm"); a gaze region is
// the narrow band the iris ratio actually moves through.
//
// x_min > x_max (or y_min > y_max) is allowed and flips that axis; the
// calibration fit produces this when the input runs opposite to the screen.
struct InputRegion {
    double x_min;
    double y_min;
    double x_max;
    double y_max;
};

inline constexpr double kMinRegionSpan = 0.02;

// Finite corners, each axis spanning at least kMinRegionSpan, corners in [-1, 2].
bool is_valid_region(const InputRegion& region) noexcept;
bool is_valid_rect(const ScreenRect& rect) noexcept;
bool contains(const ScreenRect& rect, PixelPoint point) noexcept;
PixelPoint clamp_to_rect(const ScreenRect& rect, PixelPoint point) noexcept;

// Linear interpolation from the input region to [0, 1] screen fractions,
// clamped so input outside the region pins the cursor to the screen edge.
NormalizedPoint region_to_unit(const InputRegion& region, NormalizedPoint input) noexcept;

// [0, 1] fractions to a pixel inside `target` (0 -> first pixel, 1 -> last pixel).
PixelPoint unit_to_pixel(const ScreenRect& target, NormalizedPoint unit) noexcept;

// Pixel to SendInput's 0..65535 virtual-desktop coordinates. Windows converts
// back with pixel = left + floor(absolute * width / 65536); this rounds up so
// that conversion lands on exactly the requested pixel.
AbsolutePoint pixel_to_absolute(const ScreenRect& virtual_desktop, PixelPoint pixel) noexcept;

// The inverse Windows applies. Used by tests and the live accuracy check.
PixelPoint absolute_to_pixel(const ScreenRect& virtual_desktop, AbsolutePoint absolute) noexcept;

class ScreenMapper {
public:
    // Throws std::invalid_argument when either rectangle is empty or the
    // target does not intersect the virtual desktop.
    explicit ScreenMapper(ScreenGeometry geometry);

    [[nodiscard]] PixelPoint map(const InputRegion& region, NormalizedPoint input) const noexcept;
    [[nodiscard]] AbsolutePoint to_absolute(PixelPoint pixel) const noexcept;
    [[nodiscard]] PixelPoint clamp(PixelPoint pixel) const noexcept;
    [[nodiscard]] const ScreenGeometry& geometry() const noexcept { return geometry_; }

private:
    ScreenGeometry geometry_;
};

} // namespace osi
