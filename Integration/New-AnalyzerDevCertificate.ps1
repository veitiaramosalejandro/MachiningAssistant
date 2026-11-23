param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "..\certs"),
    [string]$CommonName = "localhost"
)

$ErrorActionPreference = "Stop"
$output = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $output -Force | Out-Null

$certificate = New-SelfSignedCertificate `
    -DnsName @($CommonName, "127.0.0.1") `
    -CertStoreLocation "Cert:\CurrentUser\My" `
    -FriendlyName "MachiningAssistant Analyzer development TLS" `
    -KeyAlgorithm RSA `
    -KeyLength 2048 `
    -HashAlgorithm SHA256 `
    -NotAfter (Get-Date).AddDays(30)

try {
    $certificateBytes = $certificate.Export([Security.Cryptography.X509Certificates.X509ContentType]::Cert)
    $fingerprint = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($certificateBytes))
    $certificatePem = "-----BEGIN CERTIFICATE-----`n" +
        [Convert]::ToBase64String($certificateBytes, [Base64FormattingOptions]::InsertLineBreaks) +
        "`n-----END CERTIFICATE-----`n"
    [IO.File]::WriteAllText((Join-Path $output "machining-assistant-analyzer.crt"), $certificatePem, [Text.Encoding]::ASCII)

    $rsa = [Security.Cryptography.X509Certificates.RSACertificateExtensions]::GetRSAPrivateKey($certificate)
    if ($null -eq $rsa) { throw "No se pudo obtener la clave privada del certificado." }
    try {
        $keyPem = $rsa.ExportPkcs8PrivateKeyPem()
        [IO.File]::WriteAllText((Join-Path $output "machining-assistant-analyzer.key"), $keyPem, [Text.Encoding]::ASCII)
    }
    finally { $rsa.Dispose() }
}
finally {
    $certificate.Dispose()
}

Write-Host "Certificado TLS de desarrollo creado en: $output"
Write-Host "SHA-256 para el campo del Analyzer: $fingerprint"
Write-Host "No es un certificado de producción ni debe exponerse fuera de la red local."
