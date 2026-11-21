#include <pch.h>
#include <Config.h>

#include "InputDump_Dx12.h"

#include <fstream>
#include <magic_enum.hpp>
#include <misc/DeferredRelease.h>

#include <cmath>
#include <limits>

namespace
{
constexpr uint32_t DdsMagic = 0x20534444; // "DDS "

// DDS_HEADER, a DDS_HEADER_DXT10 follows it so the real DXGI format is kept
struct alignas(4) DdsPixelFormat
{
    uint32_t Size;
    uint32_t Flags;
    uint32_t FourCC;
    uint32_t RGBBitCount;
    uint32_t RBitMask;
    uint32_t GBitMask;
    uint32_t BBitMask;
    uint32_t ABitMask;
};

struct alignas(4) DdsHeader
{
    uint32_t Size;
    uint32_t Flags;
    uint32_t Height;
    uint32_t Width;
    uint32_t PitchOrLinearSize;
    uint32_t Depth;
    uint32_t MipMapCount;
    uint32_t Reserved1[11];
    DdsPixelFormat PixelFormat;
    uint32_t Caps;
    uint32_t Caps2;
    uint32_t Caps3;
    uint32_t Caps4;
    uint32_t Reserved2;
};

struct alignas(4) DdsHeaderDx10
{
    uint32_t Format; // DXGI_FORMAT
    uint32_t ResourceDimension;
    uint32_t MiscFlag;
    uint32_t ArraySize;
    uint32_t MiscFlags2;
};

static_assert(sizeof(DdsPixelFormat) == 32);
static_assert(sizeof(DdsHeader) == 124);
static_assert(sizeof(DdsHeaderDx10) == 20);

constexpr uint32_t DdsdCaps = 0x1;
constexpr uint32_t DdsdHeight = 0x2;
constexpr uint32_t DdsdWidth = 0x4;
constexpr uint32_t DdsdPitch = 0x8;
constexpr uint32_t DdsdPixelFormat = 0x1000;
constexpr uint32_t DdpfFourCC = 0x4;
constexpr uint32_t DdsCapsTexture = 0x1000;
constexpr uint32_t DdsFourCCDx10 = 0x30315844; // "DX10"
constexpr uint32_t D3d12Texture2D = 3;         // D3D12_RESOURCE_DIMENSION_TEXTURE2D

// A typeless color format can't be described by the DDS header, the DDS gets the typed format with the same bits
DXGI_FORMAT DdsFormat(DXGI_FORMAT InFormat)
{
    switch (InFormat)
    {
    case DXGI_FORMAT_R32_TYPELESS:
        return DXGI_FORMAT_R32_FLOAT;

    case DXGI_FORMAT_R16_TYPELESS:
        return DXGI_FORMAT_R16_FLOAT;

    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
        return DXGI_FORMAT_R10G10B10A2_UNORM;

    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        return DXGI_FORMAT_R8G8B8A8_UNORM;

    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        return DXGI_FORMAT_B8G8R8A8_UNORM;

    case DXGI_FORMAT_B8G8R8X8_TYPELESS:
        return DXGI_FORMAT_B8G8R8X8_UNORM;

    case DXGI_FORMAT_R32G32_TYPELESS:
        return DXGI_FORMAT_R32G32_FLOAT;

    case DXGI_FORMAT_R32G32B32A32_TYPELESS:
        return DXGI_FORMAT_R32G32B32A32_FLOAT;

    case DXGI_FORMAT_R16G16_TYPELESS:
        return DXGI_FORMAT_R16G16_FLOAT;

    case DXGI_FORMAT_R16G16B16A16_TYPELESS:
        return DXGI_FORMAT_R16G16B16A16_FLOAT;

    case DXGI_FORMAT_R8G8_TYPELESS:
        return DXGI_FORMAT_R8G8_UNORM;

    case DXGI_FORMAT_R8_TYPELESS:
        return DXGI_FORMAT_R8_UNORM;

    default:
        return InFormat; // Already typed or unknown
    }
}

// 16-bit float to 32-bit float, the readback of a half float format has no CPU readable type
float HalfToFloat(uint16_t InValue)
{
    const uint32_t sign = (uint32_t) (InValue & 0x8000) << 16;
    uint32_t exponent = (InValue >> 10) & 0x1F;
    uint32_t mantissa = InValue & 0x3FF;
    uint32_t bits = 0;

    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            bits = sign;
        }
        else
        {
            // Subnormal half, normalize it before shifting into a float
            exponent = 1;

            while ((mantissa & 0x400) == 0)
            {
                mantissa <<= 1;
                exponent--;
            }

            mantissa &= 0x3FF;
            bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
        }
    }
    else if (exponent == 31)
    {
        bits = sign | 0x7F800000 | (mantissa << 13); // Infinity and NaN
    }
    else
    {
        bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }

    float result = 0.0f;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

// A viewable 8-bit value, the preview is only for looking at the dump, the DDS keeps the real data
uint8_t ToneMap(float InValue)
{
    if (!std::isfinite(InValue))
        return InValue > 0.0f ? 255 : 0; // Infinity is a bright pixel, NaN is skipped

    const float value = std::clamp(InValue, 0.0f, 16.0f);
    return (uint8_t) (std::pow(value / (1.0f + value), 1.0f / 2.2f) * 255.0f + 0.5f);
}

struct DumpRange
{
    float Min = 0.0f;
    float Max = 0.0f;
    float Average = 0.0f;
};

// A range of 0..0 means the copy didn't get the texture, a small range means the data is just very dark
DumpRange GetRange(const std::vector<uint8_t>& InRows, uint32_t InWidth, uint32_t InHeight, DXGI_FORMAT InFormat)
{
    DumpRange range = { std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest(), 0.0f };
    double sum = 0.0;
    uint64_t count = 0;

    if (InFormat == DXGI_FORMAT_R16G16B16A16_FLOAT)
    {
        const auto* pixels = (const uint16_t*) InRows.data();

        for (uint64_t i = 0; i < (uint64_t) InWidth * InHeight; i++)
        {
            for (uint32_t channel = 0; channel < 3; channel++)
            {
                const float value = HalfToFloat(pixels[i * 4 + channel]);

                if (!std::isfinite(value))
                    continue;

                range.Min = std::min(range.Min, value);
                range.Max = std::max(range.Max, value);
                sum += value;
                count++;
            }
        }
    }
    else if (InFormat == DXGI_FORMAT_R16G16_FLOAT)
    {
        const auto* pixels = (const uint16_t*) InRows.data();

        for (uint64_t i = 0; i < (uint64_t) InWidth * InHeight; i++)
        {
            for (uint32_t channel = 0; channel < 2; channel++)
            {
                const float value = HalfToFloat(pixels[i * 2 + channel]);

                if (!std::isfinite(value))
                    continue;

                range.Min = std::min(range.Min, value);
                range.Max = std::max(range.Max, value);
                sum += value;
                count++;
            }
        }
    }
    else if (InFormat == DXGI_FORMAT_R32_FLOAT)
    {
        const auto* pixels = (const float*) InRows.data();

        for (uint64_t i = 0; i < (uint64_t) InWidth * InHeight; i++)
        {
            const float value = pixels[i];

            if (!std::isfinite(value))
                continue;

            range.Min = std::min(range.Min, value);
            range.Max = std::max(range.Max, value);
            sum += value;
            count++;
        }
    }
    else
    {
        range.Min = (float) *std::min_element(InRows.begin(), InRows.end());
        range.Max = (float) *std::max_element(InRows.begin(), InRows.end());
        range.Average = (range.Min + range.Max) * 0.5f;
        return range;
    }

    // Everything was NaN or the texture was empty
    if (count == 0)
    {
        range.Min = 0.0f;
        range.Max = 0.0f;
        range.Average = 0.0f;
        return range;
    }

    range.Average = (float) (sum / count);
    return range;
}

// Depth becomes R32_FLOAT so every DDS tool can open it. A depth-stencil copy is the depth plane only, so the
// stencil of a bigger format isn't in the readback, and 16/24-bit UNORM depth is scaled to 0..1
void DepthRow(const uint8_t* InSource, float* OutDepth, uint32_t InWidth, DXGI_FORMAT InFormat,
              uint32_t InSourceBytesPerPixel)
{
    switch (InFormat)
    {
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
        for (uint32_t x = 0; x < InWidth; x++)
            memcpy(&OutDepth[x], InSource + (size_t) x * InSourceBytesPerPixel, 4);
        break;

    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
        for (uint32_t x = 0; x < InWidth; x++)
        {
            uint32_t depth = 0;
            memcpy(&depth, InSource + (size_t) x * InSourceBytesPerPixel, 4);
            OutDepth[x] = (depth & 0xFFFFFF) / 16777215.0f;
        }
        break;

    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_D16_UNORM:
        for (uint32_t x = 0; x < InWidth; x++)
        {
            uint16_t depth = 0;
            memcpy(&depth, InSource + (size_t) x * InSourceBytesPerPixel, 2);
            OutDepth[x] = depth / 65535.0f;
        }
        break;

    default:
        for (uint32_t x = 0; x < InWidth; x++)
            OutDepth[x] = 0.0f;
        break;
    }
}
#pragma pack(push, 1)
struct BmpFileHeader
{
    uint16_t Type;
    uint32_t Size;
    uint16_t Reserved1;
    uint16_t Reserved2;
    uint32_t OffBits;
};

struct BmpInfoHeader
{
    uint32_t Size;
    int32_t Width;
    int32_t Height;
    uint16_t Planes;
    uint16_t BitCount;
    uint32_t Compression;
    uint32_t SizeImage;
    int32_t XPelsPerMeter;
    int32_t YPelsPerMeter;
    uint32_t ClrUsed;
    uint32_t ClrImportant;
};
#pragma pack(pop)

static_assert(sizeof(BmpFileHeader) == 14);
static_assert(sizeof(BmpInfoHeader) == 40);

// A tonemapped 8-bit preview next to the DDS, the DDS keeps the real float data
void WritePreview(const std::filesystem::path& InPath, const std::vector<uint8_t>& InRows, uint32_t InWidth,
                  uint32_t InHeight, DXGI_FORMAT InFormat, float InMin, float InMax, float InExposure)
{
    const bool halfColor = InFormat == DXGI_FORMAT_R16G16B16A16_FLOAT;
    const bool gray = InFormat == DXGI_FORMAT_R32_FLOAT;

    if (!halfColor && !gray)
        return;

    const uint32_t pixelBytes = halfColor ? 8 : 4;

    // A reversed Z depth can be almost black without stretching it to its own range
    const float depthRange = InMax > InMin ? InMax - InMin : 1.0f;
    const uint32_t rowBytes = (InWidth * 3 + 3) & ~3u;

    BmpFileHeader fileHeader = {};
    fileHeader.Type = 0x4D42; // "BM"
    fileHeader.OffBits = sizeof(BmpFileHeader) + sizeof(BmpInfoHeader);
    fileHeader.Size = fileHeader.OffBits + rowBytes * InHeight;

    BmpInfoHeader infoHeader = {};
    infoHeader.Size = sizeof(BmpInfoHeader);
    infoHeader.Width = (int32_t) InWidth;
    infoHeader.Height = (int32_t) InHeight; // Bottom-up rows
    infoHeader.Planes = 1;
    infoHeader.BitCount = 24;
    infoHeader.SizeImage = rowBytes * InHeight;

    std::ofstream stream(InPath, std::ios::binary | std::ios::trunc);

    if (!stream.is_open())
    {
        LOG_ERROR("Can't create {}", wstring_to_string(InPath.wstring()));
        return;
    }

    stream.write((const char*) &fileHeader, sizeof(fileHeader));
    stream.write((const char*) &infoHeader, sizeof(infoHeader));

    std::vector<uint8_t> row(rowBytes, 0);

    for (uint32_t y = InHeight; y > 0; y--)
    {
        const auto* source = InRows.data() + (size_t) (y - 1) * InWidth * pixelBytes;
        std::fill(row.begin(), row.end(), 0);

        for (uint32_t x = 0; x < InWidth; x++)
        {
            uint8_t red = 0;
            uint8_t green = 0;
            uint8_t blue = 0;

            if (halfColor)
            {
                const auto* pixels = (const uint16_t*) source;
                red = ToneMap(HalfToFloat(pixels[x * 4 + 0]) * InExposure);
                green = ToneMap(HalfToFloat(pixels[x * 4 + 1]) * InExposure);
                blue = ToneMap(HalfToFloat(pixels[x * 4 + 2]) * InExposure);
            }
            else
            {
                const auto* pixels = (const float*) source;
                const float value = std::clamp((pixels[x] - InMin) / depthRange, 0.0f, 1.0f);
                green = red = blue = (uint8_t) (value * 255.0f + 0.5f);
            }

            row[x * 3 + 0] = blue;
            row[x * 3 + 1] = green;
            row[x * 3 + 2] = red;
        }

        stream.write((const char*) row.data(), (std::streamsize) row.size());
    }
}
} // namespace

InputDump_Dx12::~InputDump_Dx12()
{
    if (State::Instance().isShuttingDown)
        return;

    // The GPU can still read the last frame's buffers
    for (auto& pending : _pending)
        DeferredRelease::Release(pending.Readback);

    _pending.clear();
}

bool InputDump_Dx12::NeedsFrame()
{
    // A hotkey requests a new dump even when the option stays enabled
    auto& state = State::Instance();

    if (state.dumpRequest != _lastDumpRequest)
    {
        _lastDumpRequest = state.dumpRequest;
        _dumpedFrames = 0;
    }

    auto& cfg = *Config::Instance();
    const bool enabled = cfg.DumpEnabled.value_or_default();

    // A newly enabled dump starts from its first frame again
    if (!enabled || !_wasEnabled)
        _dumpedFrames = 0;

    _wasEnabled = enabled;

    return enabled && _dumpedFrames < (uint32_t) std::max(cfg.DumpFrames.value_or_default(), 1);
}

bool InputDump_Dx12::BeginFrame(const std::filesystem::path& InFolder, const std::string& InInfo, uint32_t InFrameCount)
{
    _active = false;

    if (!NeedsFrame())
        return false;

    auto& cfg = *Config::Instance();

    std::error_code error;
    std::filesystem::create_directories(InFolder, error);

    if (error)
    {
        LOG_ERROR("Can't create {}, dumps are disabled: {}", wstring_to_string(InFolder.wstring()), error.message());
        cfg.DumpEnabled.set_volatile_value(false);
        return false;
    }

    _frameFolder = InFolder;
    _frameCount = InFrameCount;

    std::ofstream info(InFolder / "frame.txt", std::ios::binary | std::ios::trunc);

    if (info.is_open())
        info << InInfo;

    _dumpedFrames++;
    _active = true;

    LOG_INFO("Dumping upscaler inputs to {}", wstring_to_string(InFolder.wstring()));

    return true;
}

void InputDump_Dx12::Capture(ID3D12GraphicsCommandList* InCommandList, ID3D12Device* InDevice, const char* InName,
                             ID3D12Resource* InResource, D3D12_RESOURCE_STATES InState, const Subrect& InRect,
                             bool InDepth)
{
    if (!_active || InCommandList == nullptr || InDevice == nullptr || InResource == nullptr)
        return;

    const auto desc = InResource->GetDesc();

    // The upscaler's inputs are single sample 2D textures
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count > 1 ||
        desc.DepthOrArraySize > 1 || desc.MipLevels > 1)
    {
        LOG_WARN("Skipping {} for the dump, {}x{} samples {} arrays {} mips {}", InName, desc.Width, desc.Height,
                 desc.SampleDesc.Count, desc.DepthOrArraySize, desc.MipLevels);
        return;
    }

    uint32_t left = InRect.X;
    uint32_t top = InRect.Y;
    uint32_t width = InRect.Width;
    uint32_t height = InRect.Height;

    // No rect means the whole resource
    if (width == 0 || height == 0)
    {
        left = 0;
        top = 0;
        width = (uint32_t) desc.Width;
        height = desc.Height;
    }

    // Depth resources can only be copied as a whole subresource and their typeless formats are depth ones
    const bool depthResource = InDepth || (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0;

    if (depthResource)
    {
        left = 0;
        top = 0;
        width = (uint32_t) desc.Width;
        height = desc.Height;
    }

    if ((uint64_t) left + width > desc.Width || (uint64_t) top + height > desc.Height)
    {
        LOG_WARN("Skipping {} for the dump, the {}x{} view at {},{} doesn't fit {}x{}", InName, width, height, left,
                 top, desc.Width, desc.Height);
        return;
    }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
    UINT rows = 0;
    UINT64 rowSize = 0;
    UINT64 totalBytes = 0;
    InDevice->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &rowSize, &totalBytes);

    // Block compressed and other packed formats have no plain pixel size to row copy
    if (desc.Width == 0 || rowSize % desc.Width != 0)
    {
        LOG_WARN("Skipping {} for the dump, its format has no plain pixel size", InName);
        return;
    }

    const auto bytesPerPixel = (uint32_t) (rowSize / desc.Width);

    // Depth is written as R32_FLOAT, so the stencil and the extra bits of a bigger format are dropped
    const auto ddsFormat = depthResource ? DXGI_FORMAT_R32_FLOAT : DdsFormat(desc.Format);
    const auto ddsBytesPerPixel = depthResource ? 4u : bytesPerPixel;

    // The copy writes the view tightly, the footprint keeps the resource's row pitch
    footprint.Footprint.Width = width;
    footprint.Footprint.Height = height;
    footprint.Footprint.Depth = 1;

    LOG_DEBUG("Dumping {}: {}x{} view at {},{} of the {}x{} {} resource, state {}", InName, width, height, left, top,
              desc.Width, desc.Height, magic_enum::enum_name(desc.Format), (uint32_t) InState);

    D3D12_HEAP_PROPERTIES heapProperties = {};
    heapProperties.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = totalBytes;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ID3D12Resource* readback = nullptr;
    auto hr = InDevice->CreateCommittedResource(&heapProperties, D3D12_HEAP_FLAG_NONE, &bufferDesc,
                                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback));

    if (FAILED(hr))
    {
        LOG_ERROR("Can't create the readback buffer for {}, {:#x}", InName, (uint32_t) hr);
        return;
    }

    const auto transition = [InCommandList, InResource](D3D12_RESOURCE_STATES InBefore, D3D12_RESOURCE_STATES InAfter)
    {
        if (InBefore == InAfter)
            return;

        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = InResource;
        barrier.Transition.StateBefore = InBefore;
        barrier.Transition.StateAfter = InAfter;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        InCommandList->ResourceBarrier(1, &barrier);
    };

    // The upscaler keeps the input in the state the backend left it, the dump must not change that
    transition(InState, D3D12_RESOURCE_STATE_COPY_SOURCE);

    D3D12_TEXTURE_COPY_LOCATION source = {};
    source.pResource = InResource;
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    source.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION destination = {};
    destination.pResource = readback;
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint = footprint;

    D3D12_BOX box = { left, top, 0, left + width, top + height, 1 };
    InCommandList->CopyTextureRegion(&destination, 0, 0, 0, &source, depthResource ? nullptr : &box);

    transition(D3D12_RESOURCE_STATE_COPY_SOURCE, InState);

    Pending pending = {};
    pending.Path = _frameFolder / (std::string(InName) + ".dds");
    pending.Readback = readback;
    pending.Footprint = footprint;
    pending.Width = width;
    pending.Height = height;
    pending.DdsBytesPerPixel = ddsBytesPerPixel;
    pending.SourceBytesPerPixel = bytesPerPixel;
    pending.SourceFormat = desc.Format;
    pending.DdsFormat = ddsFormat;
    pending.FrameCount = _frameCount;
    pending.Depth = depthResource;
    _pending.emplace_back(pending);

    std::ofstream info(_frameFolder / "frame.txt", std::ios::binary | std::ios::app);

    if (info.is_open())
    {
        info << std::format("{}: {}x{} view at {},{} source {} dds {} pitch {}\n", InName, width, height, left, top,
                            magic_enum::enum_name(desc.Format), magic_enum::enum_name(pending.DdsFormat),
                            footprint.Footprint.RowPitch);
    }
}

void InputDump_Dx12::Flush(uint32_t InFrameCount)
{
    if (_pending.empty())
        return;

    std::vector<Pending> pending;
    pending.reserve(_pending.size());

    for (auto& entry : _pending)
    {
        // The GPU may still be writing a copy of the last frames, color alone is several MB. The data is only read
        // once three frames passed, so a torn readback can't end up in the file.
        if (InFrameCount < entry.FrameCount + 3)
        {
            pending.emplace_back(entry);
            continue;
        }

        Write(entry);
        DeferredRelease::Release(entry.Readback);
    }

    _pending = std::move(pending);
}

void InputDump_Dx12::Write(const Pending& InPending)
{
    std::ofstream stream(InPending.Path, std::ios::binary | std::ios::trunc);

    if (!stream.is_open())
    {
        LOG_ERROR("Can't create {}", wstring_to_string(InPending.Path.wstring()));
        return;
    }

    const uint32_t rowBytes = InPending.Width * InPending.DdsBytesPerPixel;

    DdsHeader header = {};
    header.Size = sizeof(DdsHeader);
    header.Flags = DdsdCaps | DdsdHeight | DdsdWidth | DdsdPitch | DdsdPixelFormat;
    header.Height = InPending.Height;
    header.Width = InPending.Width;
    header.PitchOrLinearSize = rowBytes;
    header.PixelFormat.Size = sizeof(DdsPixelFormat);
    header.PixelFormat.Flags = DdpfFourCC;
    header.PixelFormat.FourCC = DdsFourCCDx10;
    header.Caps = DdsCapsTexture;

    DdsHeaderDx10 dx10 = {};
    dx10.Format = (uint32_t) InPending.DdsFormat;
    dx10.ResourceDimension = D3d12Texture2D;
    dx10.ArraySize = 1;

    stream.write((const char*) &DdsMagic, sizeof(DdsMagic));
    stream.write((const char*) &header, sizeof(header));
    stream.write((const char*) &dx10, sizeof(dx10));

    D3D12_RANGE readRange = { 0, 0 };
    void* data = nullptr;

    if (FAILED(InPending.Readback->Map(0, &readRange, &data)))
    {
        LOG_ERROR("Can't map the readback of {}", wstring_to_string(InPending.Path.wstring()));
        return;
    }

    const auto& footprint = InPending.Footprint;
    std::vector<uint8_t> rows((size_t) rowBytes * InPending.Height);

    for (uint32_t y = 0; y < InPending.Height; y++)
    {
        const auto source = (const uint8_t*) data + footprint.Offset + (uint64_t) y * footprint.Footprint.RowPitch;
        auto destination = rows.data() + (size_t) y * rowBytes;

        if (InPending.Depth)
            DepthRow(source, (float*) destination, InPending.Width, InPending.SourceFormat,
                     InPending.SourceBytesPerPixel);
        else
            memcpy(destination, source, rowBytes);
    }

    stream.write((const char*) rows.data(), (std::streamsize) rows.size());

    D3D12_RANGE writtenRange = { 0, 0 };
    InPending.Readback->Unmap(0, &writtenRange);

    const auto written = (uint64_t) stream.tellp();
    stream.close();

    const auto range = GetRange(rows, InPending.Width, InPending.Height, InPending.DdsFormat);

    // The preview uses a simple auto exposure, a dark or bright HDR frame still shows its content
    const float exposure = range.Average > 0.0001f ? std::clamp(0.18f / range.Average, 0.0001f, 100.0f) : 1.0f;

    // The preview is for looking at the dump, the DDS keeps the real data
    if (InPending.Depth || InPending.DdsFormat == DXGI_FORMAT_R16G16B16A16_FLOAT)
    {
        auto previewPath = InPending.Path;
        previewPath.replace_extension(".bmp");
        WritePreview(previewPath, rows, InPending.Width, InPending.Height, InPending.DdsFormat, range.Min, range.Max,
                     exposure);
    }

    LOG_INFO("Dumped {} ({} bytes, {}, {:.4f}..{:.4f} avg {:.4f})", wstring_to_string(InPending.Path.wstring()),
             written, magic_enum::enum_name(InPending.DdsFormat), range.Min, range.Max, range.Average);
}
