module.exports = {
  apps: [
    {
      name: "ML-Pipeline-Platform",

      cwd: "/home/serverguy/apps/Datalect_POC/AI_POC/ML_Pipeline/ML_Pipeline_platform-O",

      script: "/home/serverguy/apps/Datalect_POC/AI_POC/venv/bin/python",

      args: "-m uvicorn backend.app.main:app --host 127.0.0.1 --port 9368",

      interpreter: "none",

      instances: 1,
      exec_mode: "fork",

      autorestart: true,
      watch: false,

      max_memory_restart: "2G",

      env: {
        PYTHONPATH:
          "/home/serverguy/apps/Datalect_POC/AI_POC/ML_Pipeline/ML_Pipeline_platform-O",
        APP_ENV: "Development"
      }
    }
  ]
};
