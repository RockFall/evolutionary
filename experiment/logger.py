import datetime

class Logger:
    @staticmethod
    def log(message):
        """
        Registra uma mensagem no console ou em um arquivo de log.
        """
        print(f"[LOG] {datetime.datetime.now()} - {message}")
