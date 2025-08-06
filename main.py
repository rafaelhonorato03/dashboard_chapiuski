#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Chapiuski - Script Principal
=====================================

Este script permite executar tanto o dashboard interativo quanto as animações.
"""

import sys
import os
import subprocess

def mostrar_menu():
    """Mostra o menu principal"""
    print("🎯 DASHBOARD CHAPIUSKI")
    print("=" * 40)
    print("1. Executar Dashboard Interativo")
    print("2. Executar Animações")
    print("3. Sair")
    print("=" * 40)

def executar_dashboard():
    """Executa o dashboard interativo"""
    print("\n🚀 Iniciando Dashboard Interativo...")
    print("O dashboard será aberto no seu navegador.")
    print("Para parar, pressione Ctrl+C")
    
    try:
        # Adicionar o diretório scripts ao path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
        
        # Importar e executar o dashboard
        from dashboard_chapiuski import app, port
        
        print(f"\n📊 Dashboard disponível em: http://localhost:{port}")
        print("Pressione Ctrl+C para parar")
        
        app.run_server(debug=True, host='0.0.0.0', port=port)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro ao executar dashboard: {e}")

def executar_animacoes():
    """Executa as animações"""
    print("\n🎬 Iniciando Animações...")
    print("As animações serão exibidas em janelas separadas.")
    print("Para parar, feche as janelas ou pressione Ctrl+C")
    
    try:
        # Executar o script de animações
        script_path = os.path.join('scripts', 'dashboard_animacoes.py')
        subprocess.run([sys.executable, script_path], check=True)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Animações interrompidas pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro ao executar animações: {e}")

def main():
    """Função principal"""
    while True:
        mostrar_menu()
        
        try:
            opcao = input("\nEscolha uma opção (1-3): ").strip()
            
            if opcao == '1':
                executar_dashboard()
            elif opcao == '2':
                executar_animacoes()
            elif opcao == '3':
                print("\n👋 Até logo!")
                break
            else:
                print("\n❌ Opção inválida! Escolha 1, 2 ou 3.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Até logo!")
            break
        except Exception as e:
            print(f"\n❌ Erro inesperado: {e}")

if __name__ == "__main__":
    main() 