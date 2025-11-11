"""
progress_bar.py - Barra de progresso limpa e profissional para experimentos de XAI

Implementa uma barra de progresso que:
- Se auto-atualiza na mesma linha (sem spam no terminal)
- Mostra informações essenciais: % | count | tempo estimado
- Compatível com PEAB, MinExp e Anchor
"""

import sys
import time
from typing import Optional


class ProgressBar:
    """
    Barra de progresso para loops de explicações.
    
    Exemplo de uso:
        progress = ProgressBar(total=100, description="Gerando explicações")
        for i in range(100):
            # ... processar item ...
            progress.update()
        progress.close()
    
    Ou usando como context manager:
        with ProgressBar(total=100, description="Gerando explicações") as pbar:
            for i in range(100):
                # ... processar item ...
                pbar.update()
    """
    
    def __init__(self, total: int, description: str = "Processando", disable: bool = False):
        """
        Args:
            total: Número total de itens a processar
            description: Texto descritivo (ex: "Gerando explicações")
            disable: Se True, desabilita a barra (útil para debugging)
        """
        self.total = total
        self.description = description
        self.disable = disable
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        if not self.disable:
            # Imprimir linha inicial
            self._render()
    
    def update(self, n: int = 1):
        """Atualiza a barra de progresso em n iterações."""
        if self.disable:
            return
        
        self.current += n
        current_time = time.time()
        
        # Atualizar no máximo a cada 0.1 segundos para não sobrecarregar
        if current_time - self.last_update_time >= 0.1 or self.current >= self.total:
            self._render()
            self.last_update_time = current_time
    
    def _render(self):
        """Renderiza a barra de progresso no terminal."""
        elapsed = time.time() - self.start_time
        
        # Calcular porcentagem
        if self.total > 0:
            percent = (self.current / self.total) * 100
        else:
            percent = 0
        
        # Calcular tempo estimado restante
        if self.current > 0:
            avg_time_per_item = elapsed / self.current
            remaining_items = self.total - self.current
            eta_seconds = avg_time_per_item * remaining_items
        else:
            eta_seconds = 0
        
        # Calcular velocidade (items/s)
        if elapsed > 0:
            speed = self.current / elapsed
        else:
            speed = 0
        
        # Formatar tempos
        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta_seconds)
        
        # Criar barra visual
        bar_length = 25
        filled_length = int(bar_length * self.current // self.total) if self.total > 0 else 0
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Montar string da barra
        progress_str = (
            f"\r{self.description}: "
            f"{percent:>5.1f}% |{bar}| "
            f"{self.current}/{self.total} "
            f"[{elapsed_str}<{eta_str}, {speed:.2f}it/s]"
        )
        
        # Escrever no terminal (sem quebra de linha)
        sys.stdout.write(progress_str)
        sys.stdout.flush()
    
    def _format_time(self, seconds: float) -> str:
        """Formata segundos em formato HH:MM:SS ou MM:SS."""
        if seconds < 0:
            return "00:00"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def close(self):
        """Finaliza a barra de progresso com quebra de linha."""
        if not self.disable:
            # Garantir que mostramos 100%
            self.current = self.total
            self._render()
            sys.stdout.write("\n")
            sys.stdout.flush()
    
    def __enter__(self):
        """Suporte para context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Suporte para context manager."""
        self.close()
        return False


def create_progress_bar(total: int, description: str = "Processando", disable: bool = False) -> ProgressBar:
    """
    Factory function para criar barra de progresso.
    
    Args:
        total: Número total de iterações
        description: Descrição do processo
        disable: Se True, desabilita a barra
    
    Returns:
        Instância de ProgressBar
    
    Exemplo:
        pbar = create_progress_bar(100, "Gerando explicações")
        for i in range(100):
            # ... processar ...
            pbar.update()
        pbar.close()
    """
    return ProgressBar(total=total, description=description, disable=disable)


# Função auxiliar para silenciar warnings verbosos de bibliotecas externas
def suppress_library_warnings():
    """
    Silencia warnings verbosos de bibliotecas como Alibi, sklearn, etc.
    Útil para manter o terminal limpo durante experimentos.
    """
    import warnings
    import logging
    
    # Suprimir warnings do Python
    warnings.filterwarnings('ignore')
    
    # Reduzir verbosidade de loggers de bibliotecas
    logging.getLogger('alibi').setLevel(logging.ERROR)
    logging.getLogger('sklearn').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    
    # Desabilitar avisos do TensorFlow se estiver instalado
    try:
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    except:
        pass


if __name__ == "__main__":
    # Teste da barra de progresso
    print("Testando barra de progresso...")
    print()
    
    # Teste 1: Progresso simples
    with ProgressBar(total=100, description="Teste 1: Rápido") as pbar:
        for i in range(100):
            time.sleep(0.01)
            pbar.update()
    
    print()
    
    # Teste 2: Progresso mais lento
    with ProgressBar(total=50, description="Teste 2: Devagar") as pbar:
        for i in range(50):
            time.sleep(0.05)
            pbar.update()
    
    print()
    print("Testes concluídos!")
