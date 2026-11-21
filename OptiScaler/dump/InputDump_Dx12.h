#pragma once
#include <d3d12.h>

#include <filesystem>
#include <string>
#include <vector>

#include <upscalers/IFeature.h>

// Copies upscaler inputs through a readback buffer and writes them as lossless DDS files.
// The copy is recorded on the game's command list right before the upscaler reads the input, the files are
// written on a later frame so the render thread never waits for the GPU.
class InputDump_Dx12
{
  private:
    struct Pending
    {
        std::filesystem::path Path;
        ID3D12Resource* Readback = nullptr;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT Footprint {};
        uint32_t Width = 0;
        uint32_t Height = 0;
        uint32_t SourceBytesPerPixel = 0;
        uint32_t DdsBytesPerPixel = 0;
        DXGI_FORMAT SourceFormat = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT DdsFormat = DXGI_FORMAT_UNKNOWN;
        uint32_t FrameCount = 0;
        bool Depth = false;
    };

    std::vector<Pending> _pending;
    std::filesystem::path _frameFolder;
    uint32_t _dumpedFrames = 0;
    uint32_t _frameCount = 0;
    uint32_t _lastDumpRequest = 0;
    bool _wasEnabled = false;
    bool _active = false;

    static void Write(const Pending& InPending);

  public:
    ~InputDump_Dx12();

    // True while the config asks for more frames, so a frame that isn't dumped doesn't build its info
    bool NeedsFrame();

    // Starts the dump when the config asks for it, the frame's folder and info file are created here
    bool BeginFrame(const std::filesystem::path& InFolder, const std::string& InInfo, uint32_t InFrameCount);

    // Records a readback copy of the part of the resource the upscaler reads, an empty rect dumps all of it
    void Capture(ID3D12GraphicsCommandList* InCommandList, ID3D12Device* InDevice, const char* InName,
                 ID3D12Resource* InResource, D3D12_RESOURCE_STATES InState, const Subrect& InRect,
                 bool InDepth = false);

    // Writes the copies the GPU finished, a copy is only read a few frames after it was recorded
    void Flush(uint32_t InFrameCount);
};
