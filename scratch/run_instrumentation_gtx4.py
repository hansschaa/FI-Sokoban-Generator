import os
import subprocess
import sys

def patch_file(filepath, replacements):
    with open(filepath, "r") as f:
        code = f.read()
    
    if "#include <chrono>" not in code:
        code = code.replace("#include <iostream>", "#include <iostream>\n#include <chrono>")
        code = code.replace("#include <vector>", "#include <vector>\n#include <chrono>")
        
    for target, rep in replacements:
        if target in code:
            code = code.replace(target, rep, 1) # ONLY REPLACE ONCE
        
    with open(filepath, "w") as f:
        f.write(code)

def main():
    print("--- 1. Inyectando timestamps (Phases A, B, C) en el codigo fuente ---")
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..") # Ensure root

    # Revert to HEAD first
    subprocess.run(["git", "restore", "src/main_experiment.cpp", "src/evolution/algorithms/genetic_algorithm.cpp", "src/evolution/evaluator/evaluator.cpp"])

    patch_file("src/main_experiment.cpp", [
        ("Individual first_valid;\n    bool found_first = false;\n\n    for (int i = 0; i < pop_size; i++)",
         "auto t_init_start = std::chrono::high_resolution_clock::now();\n    Individual first_valid;\n    bool found_first = false;\n\n    for (int i = 0; i < pop_size; i++)"),
        ("Individual best;\n    \n    // Calcular deadlock mask",
         "auto t_init_end = std::chrono::high_resolution_clock::now();\n    std::cerr << \"\\n[TIMING_PHASE] (a) Generacion Poblacion Inicial: \" << std::chrono::duration<double, std::milli>(t_init_end - t_init_start).count() << \" ms\\n\";\n    Individual best;\n    \n    // Calcular deadlock mask")
    ])

    patch_file("src/evolution/algorithms/genetic_algorithm.cpp", [
        ("        while (batch_to_evaluate.size() < offspringSize && totalAttempts < maxAttempts)",
         "        auto t_mut_start = std::chrono::high_resolution_clock::now();\n        while (batch_to_evaluate.size() < offspringSize && totalAttempts < maxAttempts)"),
        ("        // PARALLEL EVALUATION OF BATCH\n        if (!batch_to_evaluate.empty()) {",
         "        auto t_mut_end = std::chrono::high_resolution_clock::now();\n        std::cerr << \"[TIMING_PHASE] (b.1) Generacion C++ de \" << batch_to_evaluate.size() << \" mutaciones: \" << std::chrono::duration<double, std::milli>(t_mut_end - t_mut_start).count() << \" ms\\n\";\n        // PARALLEL EVALUATION OF BATCH\n        if (!batch_to_evaluate.empty()) {")
    ])

    patch_file("src/evolution/evaluator/evaluator.cpp", [
        ('auto t_flask_0 = high_resolution_clock::now();\n        auto t_fs_flask_0 = high_resolution_clock::now();',
         'auto t_flask_0 = high_resolution_clock::now();\n        auto t_fs_flask_0 = high_resolution_clock::now();\n        std::cerr << "[TIMING_PHASE] (b.2) POST Request a Flask (" << surrogate_indices.size() << " tableros)...\\n";'),
        ('auto res = cli.Post("/evaluate", payload.dump(), "application/json");\n    auto t_fs_flask_1 = high_resolution_clock::now();',
         'auto res = cli.Post("/evaluate", payload.dump(), "application/json");\n    auto t_fs_flask_1 = high_resolution_clock::now();\n    std::cerr << "[TIMING_PHASE] (b.3) Respuesta Flask HTTP " << (res ? std::to_string(res->status) : "TIMEOUT") << " en: " << std::chrono::duration<double, std::milli>(t_fs_flask_1 - t_fs_flask_0).count() << " ms\\n";'),
        ('std::cerr << "Falling back to A* solver for this batch...\\n";\n        \n        (*this->surrogate_fallbacks)++;',
         'std::cerr << "[TIMING_PHASE] (c.1) ALERTA: Fallback Silencioso a A* disparado (Conexion Fallida).\\n";\n        (*this->surrogate_fallbacks)++;'),
        ('std::cerr << "Error: Python Server returned HTTP " << res->status << "\\n";\n        std::cerr << "Response: " << res->body << "\\n";\n        \n        (*this->surrogate_fallbacks)++;',
         'std::cerr << "[TIMING_PHASE] (c.1) ALERTA: Fallback Silencioso a A* disparado (HTTP " << res->status << ").\\n";\n        (*this->surrogate_fallbacks)++;'),
        ('for (size_t idx : surrogate_indices) {\n            auto t0 = high_resolution_clock::now();\n            evaluate(population[idx]);\n        }',
         'auto t_fb_start = high_resolution_clock::now();\n        for (size_t idx : surrogate_indices) {\n            evaluate(population[idx]);\n        }\n        auto t_fb_end = high_resolution_clock::now();\n        std::cerr << "[TIMING_PHASE] (c.2) Ciclo A* de Fallback (batch entero) tardo: " << std::chrono::duration<double, std::milli>(t_fb_end - t_fb_start).count() << " ms\\n";'),
        ('// Full A* check\n            evaluate(population[idx]);',
         '// Full A* check\n            auto t_del_start = high_resolution_clock::now();\n            evaluate(population[idx]);\n            auto t_del_end = high_resolution_clock::now();\n            std::cerr << "[TIMING_PHASE] (b.4) Delegacion hibrida legal (A* para " << count_boxes(population[idx].board) << " cajas): " << std::chrono::duration<double, std::milli>(t_del_end - t_del_start).count() << " ms\\n";')
    ])

    print("--- 2. Compilando experiment_runner ---")
    subprocess.run(["make", "-C", "build", "-j12", "experiment_runner"], check=True, stdout=subprocess.DEVNULL)

    print("\n--- 3. Ejecutando Semilla 44 (Full Surrogate) en Shell 5 ---")
    cmd = [
        "./build/experiment_runner",
        "GA",
        "FO6",
        "44",
        "tuning/Instances/shell_577.txt",
        "--heuristic", "full_surrogate",
        "--maxEvals", "50",
        "--timeLimit", "100"
    ]
    
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "24"
    
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    print("\n=== LOG CRUDO DE INSTRUMENTACION (SEMILLA 44) ===")
    for line in res.stderr.split('\n'):
        if "[TIMING_PHASE]" in line or "Error:" in line or "Warning:" in line:
            print(line)
            
    print("\n=================================================")
    print("Por favor enviame el texto completo entre las lineas de '='.")

if __name__ == "__main__":
    main()
