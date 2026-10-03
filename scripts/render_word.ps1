param([Parameter(Mandatory=$true)][string]$InputDocx, [Parameter(Mandatory=$true)][string]$OutputPdf)
$ErrorActionPreference = 'Stop'
$sourceDocPath = (Resolve-Path -LiteralPath $InputDocx).Path
$targetPdfPath = [System.IO.Path]::GetFullPath($OutputPdf)
$wordRenderApp = $null
$wordRenderDocument = $null
try {
    $wordRenderApp = New-Object -ComObject Word.Application
    $wordRenderApp.Visible = $false
    $wordRenderApp.DisplayAlerts = 0
    $wordRenderApp.AutomationSecurity = 3
    $wordRenderDocument = $wordRenderApp.Documents.Open($sourceDocPath, $false, $true, $false)
    $wordRenderDocument.Repaginate()
    $wordRenderDocument.ExportAsFixedFormat($targetPdfPath, 17)
    Write-Output $targetPdfPath
} finally {
    if ($null -ne $wordRenderDocument) {
        $wordRenderDocument.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordRenderDocument)
    }
    if ($null -ne $wordRenderApp) {
        $wordRenderApp.Quit(0)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordRenderApp)
    }
}
