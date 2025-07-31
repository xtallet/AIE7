# LangSmith Integration

Este proyecto incluye integración completa con LangSmith para registrar todas las trazas de las preguntas que se lanzan desde el frontend.

## Configuración

### Backend

La configuración de LangSmith se realiza en el backend de la siguiente manera:

1. **Endpoint `/ask`** (`backend/main.py`):
   - Acepta un parámetro opcional `langsmith_api_key`
   - Configura las variables de entorno de LangSmith cuando se proporciona la API key

2. **Función `run_agentic_rag`** (`backend/agent_graph.py`):
   - Configura LangSmith al inicio del procesamiento
   - Pasa la configuración a través de toda la cadena de procesamiento

### Frontend

El frontend ya incluye un campo opcional para la LangSmith API key:

- Campo de entrada: "LangSmith API Key (optional)"
- Se envía automáticamente al backend cuando se proporciona
- No es requerido para el funcionamiento básico

## Variables de Entorno Configuradas

Cuando se proporciona una LangSmith API key, se configuran las siguientes variables de entorno:

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIE7-S11-Certification-Challenge-{uuid4().hex[0:8]}"
os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
```

## Funcionamiento

### Con LangSmith
1. El usuario proporciona su LangSmith API key en el frontend
2. El backend configura LangSmith antes de procesar la pregunta
3. Todas las operaciones de LangChain se registran en LangSmith
4. Se puede ver el flujo completo en el dashboard de LangSmith

### Sin LangSmith
1. El usuario deja vacío el campo de LangSmith API key
2. El backend procesa la pregunta normalmente sin registro de trazas
3. La funcionalidad básica funciona igual

## Beneficios

- **Trazabilidad completa**: Todas las operaciones de LangChain se registran
- **Debugging mejorado**: Se puede ver exactamente qué pasó en cada pregunta
- **Optimización**: Se pueden identificar cuellos de botella y optimizar el flujo
- **Monitoreo**: Se puede hacer seguimiento del rendimiento y uso

## Uso

1. Obtén tu LangSmith API key desde [LangSmith](https://smith.langchain.com/)
2. Inicia el backend: `cd backend && python main.py`
3. Inicia el frontend: `cd frontend && npm start`
4. Sube un PDF y haz una pregunta
5. Proporciona tu LangSmith API key (opcional)
6. Ve las trazas en tu dashboard de LangSmith

## Estructura de Proyecto

```
├── backend/
│   ├── main.py              # Endpoint con configuración de LangSmith
│   └── agent_graph.py       # Lógica de procesamiento con LangSmith
├── frontend/
│   └── src/
│       └── App.js           # Interfaz con campo de LangSmith API key
└── test_langsmith_integration.py  # Script de prueba
```

## Notas Técnicas

- La configuración de LangSmith se hace al inicio del procesamiento para asegurar que todas las operaciones se registren
- Cada sesión genera un proyecto único con un UUID para evitar conflictos
- El campo de LangSmith API key es completamente opcional
- La integración es transparente para el usuario final 