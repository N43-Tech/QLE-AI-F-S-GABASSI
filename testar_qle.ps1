$checkpoint = ".\checkpoints\qle_base_300.pt"
$resultados = ".\resultados_qle_300.txt"

Remove-Item $resultados -ErrorAction SilentlyContinue

$testes = @(
    "Usuário: Qual é o seu nome?`nqLE:",
    "Usuário: Quem é você?`nqLE:",
    "Usuário: Como devo chamar você?`nqLE:",
    "Usuário: Seu nome é Tess?`nqLE:",
    "Usuário: Apresente-se em uma frase.`nqLE:",
    "Usuário: O que é inteligência artificial?`nqLE:",
    "Usuário: Explique automação industrial.`nqLE:",
    "Usuário: Quanto é 8 mais 7?`nqLE:"
)

foreach ($prompt in $testes) {
    $separador = "`n========================================"
    $cabecalho = "$separador`nPROMPT:`n$prompt`n"

    Write-Host $cabecalho
    Add-Content -Path $resultados -Value $cabecalho

    python -m src.generate `
        --checkpoint $checkpoint `
        --prompt $prompt `
        --tokens 100 `
        --temperatura 0.2 `
        --top-k 8 |
        Tee-Object -FilePath $resultados -Append
}

Write-Host "`nTestes salvos em: $resultados"