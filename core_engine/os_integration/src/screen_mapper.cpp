#include "os_integration/screen_mapper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace osi {
namespace {

constexpr std::int64_t kAbsoluteSpan = 65536;
constexpr std::int32_t kAbsoluteMax = 65535;

double clamp01(double value) noexcept {
    return std::min(1.0, std::max(0.0, value));
}

bool corner_ok(double value) noexcept {
    return std::isfinite(value) && value >= -1.0 && value <= 2.0;
}

// Maps one axis; NaN input (already excluded upstream) would clamp to 0.
double axis_to_unit(double value, double lo, double hi) noexcept {
    const double t = (value - lo) / (hi - lo);
    return std::isfinite(t) ? clamp01(t) : 0.0;
}

std::int32_t unit_to_axis(double unit, std::int32_t origin, std::int32_t length) noexcept {
    const double offset = std::round(clamp01(unit) * static_cast<double>(length - 1));
    return origin + static_cast<std::int32_t>(offset);
}

std::int32_t pixel_axis_to_absolute(std::int32_t pixel, std::int32_t origin,
                                    std::int32_t length) noexcept {
    const std::int64_t last = static_cast<std::int64_t>(length) - 1;
    const std::int64_t offset =
        std::min(last, std::max<std::int64_t>(0, static_cast<std::int64_t>(pixel) - origin));
    // ceil(offset * 65536 / length): smallest absolute value that Windows
    // floors back to `offset`.
    const std::int64_t absolute = (offset * kAbsoluteSpan + length - 1) / length;
    return static_cast<std::int32_t>(std::min<std::int64_t>(absolute, kAbsoluteMax));
}

std::int32_t absolute_axis_to_pixel(std::int32_t absolute, std::int32_t origin,
                                    std::int32_t length) noexcept {
    const std::int64_t clamped = std::min<std::int64_t>(kAbsoluteMax, std::max(0, absolute));
    return origin + static_cast<std::int32_t>(clamped * length / kAbsoluteSpan);
}

} // namespace

bool is_valid_region(const InputRegion& region) noexcept {
    return corner_ok(region.x_min) && corner_ok(region.x_max) && corner_ok(region.y_min) &&
           corner_ok(region.y_max) && std::abs(region.x_max - region.x_min) >= kMinRegionSpan &&
           std::abs(region.y_max - region.y_min) >= kMinRegionSpan;
}

bool is_valid_rect(const ScreenRect& rect) noexcept {
    return rect.width > 0 && rect.height > 0;
}

bool contains(const ScreenRect& rect, PixelPoint point) noexcept {
    const std::int64_t right = static_cast<std::int64_t>(rect.left) + rect.width;
    const std::int64_t bottom = static_cast<std::int64_t>(rect.top) + rect.height;
    return point.x >= rect.left && point.x < right && point.y >= rect.top && point.y < bottom;
}

PixelPoint clamp_to_rect(const ScreenRect& rect, PixelPoint point) noexcept {
    const std::int32_t right = rect.left + rect.width - 1;
    const std::int32_t bottom = rect.top + rect.height - 1;
    return {std::min(right, std::max(rect.left, point.x)),
            std::min(bottom, std::max(rect.top, point.y))};
}

NormalizedPoint region_to_unit(const InputRegion& region, NormalizedPoint input) noexcept {
    return {axis_to_unit(input.x, region.x_min, region.x_max),
            axis_to_unit(input.y, region.y_min, region.y_max)};
}

PixelPoint unit_to_pixel(const ScreenRect& target, NormalizedPoint unit) noexcept {
    return {unit_to_axis(unit.x, target.left, target.width),
            unit_to_axis(unit.y, target.top, target.height)};
}

AbsolutePoint pixel_to_absolute(const ScreenRect& virtual_desktop, PixelPoint pixel) noexcept {
    return {pixel_axis_to_absolute(pixel.x, virtual_desktop.left, virtual_desktop.width),
            pixel_axis_to_absolute(pixel.y, virtual_desktop.top, virtual_desktop.height)};
}

PixelPoint absolute_to_pixel(const ScreenRect& virtual_desktop, AbsolutePoint absolute) noexcept {
    return {absolute_axis_to_pixel(absolute.x, virtual_desktop.left, virtual_desktop.width),
            absolute_axis_to_pixel(absolute.y, virtual_desktop.top, virtual_desktop.height)};
}

ScreenMapper::ScreenMapper(ScreenGeometry geometry) : geometry_(geometry) {
    if (!is_valid_rect(geometry.virtual_desktop) || !is_valid_rect(geometry.target)) {
        throw std::invalid_argument("Screen rectangles must have a positive size");
    }
    // Clip the target to the desktop so every mapped pixel is reachable.
    const ScreenRect& desk = geometry.virtual_desktop;
    const std::int64_t left = std::max<std::int64_t>(geometry.target.left, desk.left);
    const std::int64_t top = std::max<std::int64_t>(geometry.target.top, desk.top);
    const std::int64_t right = std::min<std::int64_t>(
        static_cast<std::int64_t>(geometry.target.left) + geometry.target.width,
        static_cast<std::int64_t>(desk.left) + desk.width);
    const std::int64_t bottom = std::min<std::int64_t>(
        static_cast<std::int64_t>(geometry.target.top) + geometry.target.height,
        static_cast<std::int64_t>(desk.top) + desk.height);
    if (right <= left || bottom <= top) {
        throw std::invalid_argument("Target screen does not intersect the virtual desktop");
    }
    geometry_.target = {static_cast<std::int32_t>(left), static_cast<std::int32_t>(top),
                        static_cast<std::int32_t>(right - left),
                        static_cast<std::int32_t>(bottom - top)};
}

PixelPoint ScreenMapper::map(const InputRegion& region, NormalizedPoint input) const noexcept {
    return unit_to_pixel(geometry_.target, region_to_unit(region, input));
}

AbsolutePoint ScreenMapper::to_absolute(PixelPoint pixel) const noexcept {
    return pixel_to_absolute(geometry_.virtual_desktop, pixel);
}

PixelPoint ScreenMapper::clamp(PixelPoint pixel) const noexcept {
    return clamp_to_rect(geometry_.target, pixel);
}

} // namespace osi
