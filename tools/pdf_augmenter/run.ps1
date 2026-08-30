[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string] $InputPath,
    [Alias('o', 'output')]
    [string] $OutputPath,
    [switch] $Recursive,
    [string] $Seed,
    [int] $Dpi,
    [Alias('rotation-min')]
    [double] $RotationMin,
    [Alias('rotation-max')]
    [double] $RotationMax,
    [Alias('translation-min')]
    [double] $TranslationMin,
    [Alias('translation-max')]
    [double] $TranslationMax,
    [Alias('blur-min')]
    [double] $BlurMin,
    [Alias('blur-max')]
    [double] $BlurMax,
    [Alias('sharpen-min')]
    [double] $SharpenMin,
    [Alias('sharpen-max')]
    [double] $SharpenMax,
    [ValidateSet('random', 'blur', 'sharpen')]
    [string] $Effect,
    [Alias('image-format')]
    [ValidateSet('jpeg', 'png')]
    [string] $ImageFormat,
    [Alias('jpeg-quality')]
    [int] $JpegQuality,
    [Alias('max-megapixels')]
    [double] $MaxMegapixels,
    [Alias('remove-footers')]
    [switch] $RemoveFooters,
    [Alias('footer-height-percent')]
    [double] $FooterHeightPercent,
    [string] $Suffix,
    [switch] $Overwrite,
    [Alias('fail-fast')]
    [switch] $FailFast,
    [Alias('skip-verification')]
    [switch] $SkipVerification,
    [Alias('manifest')]
    [string] $ManifestPath,
    [Alias('no-manifest')]
    [switch] $NoManifest,
    [Alias('h')]
    [switch] $Help
)

$ErrorActionPreference = 'Stop'
$toolRoot = $PSScriptRoot
$venvRoot = Join-Path $toolRoot '.venv'
$pythonPath = Join-Path $venvRoot 'Scripts/python.exe'

if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        & $pyLauncher.Source -3 -m venv $venvRoot
    }
    else {
        $pythonLauncher = Get-Command python -ErrorAction Stop
        & $pythonLauncher.Source -m venv $venvRoot
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python virtual environment."
    }
}

& $pythonPath -c "import PIL, pypdf, pypdfium2, reportlab" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $pythonPath -m pip install -r (Join-Path $toolRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install PDF augmenter dependencies."
    }
}

$forwardArgs = @()
if ($Help) { $forwardArgs += '--help' }
if ($InputPath) { $forwardArgs += $InputPath }
if ($PSBoundParameters.ContainsKey('OutputPath')) { $forwardArgs += @('-o', $OutputPath) }
if ($Recursive) { $forwardArgs += '--recursive' }
if ($PSBoundParameters.ContainsKey('Seed')) { $forwardArgs += @('--seed', $Seed) }
if ($PSBoundParameters.ContainsKey('Dpi')) { $forwardArgs += @('--dpi', [string]$Dpi) }
if ($PSBoundParameters.ContainsKey('RotationMin')) { $forwardArgs += @('--rotation-min', [string]$RotationMin) }
if ($PSBoundParameters.ContainsKey('RotationMax')) { $forwardArgs += @('--rotation-max', [string]$RotationMax) }
if ($PSBoundParameters.ContainsKey('TranslationMin')) { $forwardArgs += @('--translation-min', [string]$TranslationMin) }
if ($PSBoundParameters.ContainsKey('TranslationMax')) { $forwardArgs += @('--translation-max', [string]$TranslationMax) }
if ($PSBoundParameters.ContainsKey('BlurMin')) { $forwardArgs += @('--blur-min', [string]$BlurMin) }
if ($PSBoundParameters.ContainsKey('BlurMax')) { $forwardArgs += @('--blur-max', [string]$BlurMax) }
if ($PSBoundParameters.ContainsKey('SharpenMin')) { $forwardArgs += @('--sharpen-min', [string]$SharpenMin) }
if ($PSBoundParameters.ContainsKey('SharpenMax')) { $forwardArgs += @('--sharpen-max', [string]$SharpenMax) }
if ($PSBoundParameters.ContainsKey('Effect')) { $forwardArgs += @('--effect', $Effect) }
if ($PSBoundParameters.ContainsKey('ImageFormat')) { $forwardArgs += @('--image-format', $ImageFormat) }
if ($PSBoundParameters.ContainsKey('JpegQuality')) { $forwardArgs += @('--jpeg-quality', [string]$JpegQuality) }
if ($PSBoundParameters.ContainsKey('MaxMegapixels')) { $forwardArgs += @('--max-megapixels', [string]$MaxMegapixels) }
if ($RemoveFooters) { $forwardArgs += '--remove-footers' }
if ($PSBoundParameters.ContainsKey('FooterHeightPercent')) { $forwardArgs += @('--footer-height-percent', [string]$FooterHeightPercent) }
if ($PSBoundParameters.ContainsKey('Suffix')) { $forwardArgs += @('--suffix', $Suffix) }
if ($Overwrite) { $forwardArgs += '--overwrite' }
if ($FailFast) { $forwardArgs += '--fail-fast' }
if ($SkipVerification) { $forwardArgs += '--skip-verification' }
if ($PSBoundParameters.ContainsKey('ManifestPath')) { $forwardArgs += @('--manifest', $ManifestPath) }
if ($NoManifest) { $forwardArgs += '--no-manifest' }

& $pythonPath (Join-Path $toolRoot 'pdf_augmenter.py') @forwardArgs
exit $LASTEXITCODE
