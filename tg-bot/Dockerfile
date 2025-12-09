FROM mambaorg/micromamba:1.5.8

ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /app

# Сначала копируем только окружение, чтобы кэшировать установку зависимостей
COPY --chown=micromamba:micromamba environment.yaml /tmp/environment.yaml
RUN micromamba create -y -n moodbot -f /tmp/environment.yaml && \
    micromamba clean --all --yes

# Копируем исходный код
COPY --chown=micromamba:micromamba . .

# Запуск бота внутри подготовленного окружения
CMD ["micromamba", "run", "-n", "moodbot", "python", "bot.py"]
