using Documenter
using DocumenterMermaid
using HCM

makedocs(
    sitename = "HCM.jl",
    modules = [HCM],
    pages = [
        "Home" => "index.md",
        "Vignettes" => [
            "Getting Started" => "vignettes/getting_started.md",
            "General Identification Engine" => "vignettes/identification_engine.md",
            "Interference vs. SUTVA" => "vignettes/interference_vs_sutva.md",
        ],
        "Reference" => "reference.md",
    ],
    warnonly = true,
)

deploydocs(
    repo = "github.com/xiangao/HCM.jl.git",
    devbranch = "master",
    push_preview = false,
)
