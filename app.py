"""
================================================================================
APP STREAMLIT — Predicción de Deserción Estudiantil
================================================================================
Proyecto de Minería de Datos — CRISP-DM
Carrera: Ciencia de Datos e Inteligencia Artificial

EJECUCIÓN:
    streamlit run app.py

ARCHIVOS REQUERIDOS (misma carpeta):
    modelo_desercion.pkl           Modelo entrenado
    scaler.pkl                     Escalador
    features.pkl                   Lista de features
    resultados_modelos.pkl         Métricas de modelos
    dataset_modelado.csv           Dataset procesado
    feature_importances.csv        Importancia de variables
    test_predictions.csv           Predicciones del test set
    REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.xlsx   Dataset original
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import confusion_matrix, classification_report

# ================================================================================
# CONFIGURACIÓN DE PÁGINA
# ================================================================================
st.set_page_config(
    page_title="Predicción de Deserción Estudiantil",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================================
# CARGA DE DATOS Y MODELOS (con caché)
# ================================================================================
@st.cache_data
def cargar_datos():
    """Carga el dataset original y el procesado desde los archivos CSV/Excel"""
    df_orig = pd.read_excel('REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.xlsx')
    df_orig['PROMEDIO'] = df_orig['PROMEDIO'].str.replace(',', '.').astype(float)
    df_mod = pd.read_csv('dataset_modelado.csv')
    test_pred = pd.read_csv('test_predictions.csv')
    return df_orig, df_mod, test_pred

@st.cache_resource
def cargar_modelo():
    """Carga el modelo, scaler, features y resultados desde archivos pkl"""
    modelo = joblib.load('modelo_desercion.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    resultados = joblib.load('resultados_modelos.pkl')
    importancias = pd.read_csv('feature_importances.csv', index_col=0, header=None).squeeze()
    return modelo, scaler, features, resultados, importancias

try:
    df_orig, df_mod, test_pred = cargar_datos()
    modelo, scaler, features, resultados, importancias = cargar_modelo()
    DATOS_OK = True
except Exception as e:
    DATOS_OK = False
    ERROR = str(e)

# ================================================================================
# SIDEBAR — NAVEGACIÓN
# ================================================================================
st.sidebar.title("🎓 Deserción Estudiantil")
st.sidebar.markdown("*Minería de Datos — CRISP-DM*")
st.sidebar.markdown("---")

PAGINAS = {
    "🏠 Inicio": "inicio",
    "📊 Análisis Exploratorio (EDA)": "eda",
    "🤖 Modelo y Métricas": "modelo",
    "🔮 Predicción Individual": "prediccion",
    "📋 Explorar Datos": "datos"
}
pagina = PAGINAS[st.sidebar.radio("Navegación", list(PAGINAS.keys()))]

st.sidebar.markdown("---")
st.sidebar.info("**Carrera:** Ciencia de Datos e IA\n\n**Metodología:** CRISP-DM\n\n**Modelos:** LR, RF, GB, SVM")

if not DATOS_OK:
    st.error(f"⚠️ Error al cargar archivos: {ERROR}")
    st.info("Coloca todos los archivos generados por el notebook en la misma carpeta que `app.py`.")
    st.stop()

# ================================================================================
# CONSTANTES
# ================================================================================
PERIODOS = ['2023 - 2024 CII','2023 - 2024 ING2B','2024 - 2025 CI','2024 - 2025 ING1B',
            '2024 - 2025 CII','2024 - 2025 ING2B','2025 - 2026 CI','2025 - 2026 ING1A','2025 - 2026 ING1B']
ULTIMOS = PERIODOS[6:]
n_total = len(df_mod)
n_des = int(df_mod['DESERTOR'].sum())
n_act = n_total - n_des
nombre_modelo = type(modelo).__name__

# ================================================================================
# PÁGINA: INICIO
# ================================================================================
if pagina == "inicio":
    st.title("🎓 Sistema de Predicción de Deserción Estudiantil")
    st.markdown("### Proyecto de Minería de Datos — Metodología CRISP-DM")
    st.markdown("---")

    # Dos columnas: contexto + métricas
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        #### 📌 Contexto del Problema
        La deserción estudiantil representa uno de los principales desafíos que enfrentan
        las instituciones de educación superior. Identificar de manera temprana a los
        estudiantes con mayor riesgo de abandono permite implementar estrategias de
        intervención oportunas.

        #### 🎯 Objetivo
        Desarrollar un modelo predictivo que identifique estudiantes en riesgo de deserción
        utilizando datos históricos del record académico, aplicando técnicas de minería de
        datos bajo la metodología CRISP-DM.

        #### 📝 Definición de Deserción
        Un estudiante se clasifica como **desertor** si su último período de matrícula
        es **anterior** al ciclo académico más reciente (2025-2026).
        """)

    with col2:
        st.markdown("#### 📊 Resumen del Dataset")
        st.metric("Total Estudiantes", n_total)
        m1, m2 = st.columns(2)
        m1.metric("Activos", n_act, delta=f"{n_act/n_total*100:.1f}%")
        m2.metric("Desertores", n_des, delta=f"{n_des/n_total*100:.1f}%", delta_color="inverse")
        st.metric("Registros Académicos", f"{len(df_orig):,}")
        st.metric("Materias", df_orig['MATERIA'].nunique())
        st.metric("Períodos", df_orig['PERIODO'].nunique())
        st.metric("Modelo Final", nombre_modelo)

    st.markdown("---")
    st.markdown("""
    #### 🔄 Fases CRISP-DM Aplicadas
    | Fase | Descripción | Estado |
    |------|-------------|--------|
    | 1. Comprensión del Negocio | Definición de deserción y criterios de éxito | ✅ |
    | 2. Comprensión de los Datos | EDA: 4,448 registros, 488 estudiantes, 30 materias | ✅ |
    | 3. Preparación de Datos | Feature engineering: 40+ variables → selección final | ✅ |
    | 4. Modelado | 4 algoritmos: LR, RF, GB, SVM con CV 5-fold + SMOTE | ✅ |
    | 5. Evaluación | Comparación de métricas, análisis de errores | ✅ |
    | 6. Despliegue | Esta aplicación Streamlit | ✅ |
    """)

# ================================================================================
# PÁGINA: EDA
# ================================================================================
elif pagina == "eda":
    st.title("📊 Análisis Exploratorio de Datos")
    st.markdown("### Fase 2 CRISP-DM — Comprensión de los Datos")
    st.markdown("---")

    # Métricas rápidas
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registros", f"{len(df_orig):,}")
    m2.metric("Estudiantes", df_orig['ESTUDIANTE'].nunique())
    m3.metric("Tasa Aprobación", f"{(df_orig['ESTADO']=='APROBADA').mean()*100:.1f}%")
    m4.metric("Promedio General", f"{df_orig['PROMEDIO'].mean():.2f}")
    st.markdown("---")

    # Pestañas para organizar gráficos
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribuciones", "🎯 Deserción", "📚 Materias", "🔗 Correlaciones"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_orig, x='PROMEDIO', nbins=40, color='ESTADO',
                             title='Distribución de Promedios por Estado',
                             color_discrete_map={'APROBADA':'#2ecc71','REPROBADA':'#e74c3c'},
                             barmode='overlay', opacity=0.7)
            fig.add_vline(x=7.0, line_dash="dash", line_color="orange", annotation_text="Mín. aprobación (7.0)")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(df_orig, x='ASISTENCIA', nbins=30, title='Distribución de Asistencia',
                             color_discrete_sequence=['#9b59b6'])
            fig.add_vline(x=df_orig['ASISTENCIA'].mean(), line_dash="dash", line_color="red",
                         annotation_text=f"Media: {df_orig['ASISTENCIA'].mean():.1f}%")
            st.plotly_chart(fig, use_container_width=True)

        # Estudiantes por período
        ep = df_orig.groupby('PERIODO')['ESTUDIANTE'].nunique().reindex(PERIODOS)
        fig = px.bar(x=[p.replace('2023 - 2024','23-24').replace('2024 - 2025','24-25').replace('2025 - 2026','25-26') for p in PERIODOS],
                    y=ep.values, title='Estudiantes por Período', text=ep.values,
                    labels={'x':'Período','y':'Estudiantes'})
        fig.update_traces(marker_color=['#3498db']*6+['#2ecc71']*3)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(values=[n_act, n_des], names=['Activo','Desertor'],
                        title='Variable Objetivo', color_discrete_sequence=['#2ecc71','#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            niv = df_mod.groupby('nivel_max')['DESERTOR'].mean()*100
            fig = px.bar(x=niv.index.astype(str), y=niv.values,
                        title='Tasa de Deserción por Nivel Máximo',
                        labels={'x':'Nivel','y':'% Deserción'}, text=[f'{v:.1f}%' for v in niv.values])
            fig.update_traces(marker_color=['#e74c3c' if v>30 else '#f39c12' if v>15 else '#2ecc71' for v in niv.values])
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            df_mod['Condicion'] = df_mod['DESERTOR'].map({0:'Activo',1:'Desertor'})
            fig = px.box(df_mod, x='Condicion', y='promedio_general', color='Condicion',
                        title='Promedio General por Condición',
                        color_discrete_map={'Activo':'#2ecc71','Desertor':'#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(df_mod, x='Condicion', y='asistencia_promedio', color='Condicion',
                        title='Asistencia Promedio por Condición',
                        color_discrete_map={'Activo':'#2ecc71','Desertor':'#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        tasa = df_orig.groupby('MATERIA')['ESTADO'].apply(lambda x: (x=='REPROBADA').sum()/len(x)*100).sort_values()
        fig = px.bar(x=tasa.values, y=tasa.index, orientation='h',
                    title='Tasa de Reprobación por Materia (%)', labels={'x':'% Reprobación','y':''})
        fig.update_traces(marker_color=['#e74c3c' if v>25 else '#f39c12' if v>15 else '#2ecc71' for v in tasa.values])
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        feats_corr = [f for f in features if f in df_mod.columns] + ['DESERTOR']
        corr_data = df_mod[feats_corr].corr()
        fig_mpl, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, ax=ax, annot_kws={'size':7}, vmin=-1, vmax=1)
        ax.set_title('Matriz de Correlación', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_mpl)

    with st.expander("📋 Estadísticas Descriptivas"):
        st.dataframe(df_orig[['PROMEDIO','ASISTENCIA','NO. VEZ','NIVEL']].describe().round(2))

# ================================================================================
# PÁGINA: MODELO Y MÉTRICAS
# ================================================================================
elif pagina == "modelo":
    st.title("🤖 Modelo y Métricas de Evaluación")
    st.markdown("### Fases 4 y 5 CRISP-DM — Modelado y Evaluación")
    st.markdown("---")

    # Tabla comparativa
    st.subheader("📊 Comparación de Modelos")
    df_res = pd.DataFrame(resultados).T.sort_values('F1-Score', ascending=False)
    mejor = df_res.index[0]
    st.success(f"🏆 **Mejor modelo:** {mejor} — F1-Score: {df_res.loc[mejor,'F1-Score']:.4f} | Recall: {df_res.loc[mejor,'Recall']:.4f} | AUC: {df_res.loc[mejor,'AUC-ROC']:.4f}")

    st.dataframe(
        df_res.style.highlight_max(axis=0, color='#d4edda').format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("---")

    # Gráficos lado a lado
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📈 Métricas por Modelo")
        metricas = ['Accuracy','Precision','Recall','F1-Score','AUC-ROC']
        fig = go.Figure()
        colores = ['#3498db','#2ecc71','#e67e22','#9b59b6']
        for i, (nombre, vals) in enumerate(df_res.iterrows()):
            fig.add_trace(go.Bar(name=nombre, x=metricas,
                                y=[vals[m] for m in metricas],
                                marker_color=colores[i % len(colores)],
                                text=[f'{vals[m]:.3f}' for m in metricas],
                                textposition='outside'))
        fig.update_layout(barmode='group', yaxis_range=[0, 1.15], height=450)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("🔢 Matriz de Confusión")
        y_real = test_pred['y_test'].values
        y_pred = test_pred['y_pred'].values
        cm = confusion_matrix(y_real, y_pred)

        fig_cm, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={'size': 20},
                   xticklabels=['Activo','Desertor'], yticklabels=['Activo','Desertor'])
        ax.set_ylabel('Real', fontsize=12); ax.set_xlabel('Predicho', fontsize=12)
        ax.set_title(f'Matriz de Confusión — Modelo Final', fontweight='bold')
        st.pyplot(fig_cm)

    st.markdown("---")

    # Reporte de clasificación
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📋 Reporte de Clasificación")
        report = classification_report(y_real, y_pred, target_names=['Activo','Desertor'], output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    with c2:
        st.subheader("💡 Interpretación")
        st.markdown(f"""
        - **Accuracy:** {report['accuracy']:.1%} de predicciones correctas en total
        - **Precision (Desertor):** {report['Desertor']['precision']:.1%} de los que predice como desertores realmente lo son
        - **Recall (Desertor):** {report['Desertor']['recall']:.1%} de los desertores reales son detectados
        - **F1-Score:** {report['Desertor']['f1-score']:.4f} (equilibrio precisión-recall)
        """)

    st.markdown("---")

    # Importancia de variables
    st.subheader("🏆 Variables Más Importantes para la Predicción")
    imp_sorted = importancias.sort_values(ascending=True)
    fig = px.bar(x=imp_sorted.values, y=imp_sorted.index, orientation='h',
                title='Importancia de Variables', labels={'x':'Importancia','y':''})
    fig.update_traces(marker_color='#2980b9')
    fig.update_layout(height=max(400, len(imp_sorted)*25))
    st.plotly_chart(fig, use_container_width=True)

    # Tabla de interpretación
    with st.expander("💡 ¿Qué significa cada variable?"):
        interp = {
            'tasa_reprobacion': 'Proporción de materias reprobadas sobre el total cursado',
            'materias_aprobadas': 'Cantidad total de materias aprobadas en toda la carrera',
            'promedio_general': 'Media de calificaciones de todas las materias',
            'promedio_max': 'Mejor calificación obtenida en alguna materia',
            'asistencia_promedio': 'Media del porcentaje de asistencia',
            'materias_max_periodo': 'Máximo de materias cursadas en un solo período',
            'materias_prom_periodo': 'Promedio de materias por período cursado',
            'num_periodos_regulares': 'Cantidad de períodos regulares (no inglés) cursados',
            'promedio_std': 'Variabilidad de las calificaciones (desviación estándar)',
            'asistencia_std': 'Variabilidad de la asistencia',
            'nivel_max': 'Nivel más alto alcanzado en la carrera',
            'nivel_min': 'Nivel más bajo en el que cursó materias',
            'prom_ultimo': 'Promedio de calificaciones en el último período',
            'cambio_promedio': 'Diferencia de promedio entre último y primer período',
            'cambio_asistencia': 'Diferencia de asistencia entre último y primer período',
            'registros_especiales': 'Registros de movilidad/homologación/convalidación',
        }
        for f in imp_sorted.index[::-1][:10]:
            desc = interp.get(f, f'Variable: {f}')
            st.markdown(f"- **{f}**: {desc}")

# ================================================================================
# PÁGINA: PREDICCIÓN INDIVIDUAL
# ================================================================================
elif pagina == "prediccion":
    st.title("🔮 Predicción de Riesgo de Deserción")
    st.markdown("Ingrese los datos académicos de un estudiante para obtener su nivel de riesgo.")
    st.markdown("---")

    # Formulario dividido en 3 columnas
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**📚 Rendimiento Académico**")
        v_prom_gen = st.slider("Promedio General (0-10)", 0.0, 10.0, 7.5, 0.1)
        v_prom_max = st.slider("Mejor Nota Obtenida", 0.0, 10.0, 9.0, 0.1)
        v_prom_min = st.slider("Peor Nota Obtenida", 0.0, 10.0, 5.0, 0.1)
        v_prom_std = st.slider("Variabilidad de Notas", 0.0, 5.0, 1.0, 0.1)
        v_prom_ult = st.slider("Promedio Último Período", 0.0, 10.0, 7.0, 0.1)
        v_cambio_prom = st.slider("Cambio de Promedio", -10.0, 10.0, 0.0, 0.1)

    with c2:
        st.markdown("**📋 Asistencia**")
        v_asist_prom = st.slider("Asistencia Promedio (%)", 0, 100, 85)
        v_asist_min = st.slider("Asistencia Mínima (%)", 0, 100, 50)
        v_asist_std = st.slider("Variabilidad Asistencia", 0.0, 50.0, 10.0, 0.5)
        v_asist_ult = st.slider("Asistencia Último Período (%)", 0, 100, 80)
        v_cambio_asist = st.slider("Cambio de Asistencia", -100.0, 100.0, 0.0, 1.0)

    with c3:
        st.markdown("**📊 Historial Académico**")
        v_mat_apr = st.number_input("Materias Aprobadas", 0, 30, 8)
        v_mat_rep = st.number_input("Materias Reprobadas", 0, 15, 1)
        v_total = v_mat_apr + v_mat_rep
        v_tasa_rep = v_mat_rep / v_total if v_total > 0 else 0
        st.caption(f"Tasa de reprobación calculada: {v_tasa_rep:.2%}")
        v_max_periodo = st.number_input("Máx. Materias/Período", 1, 11, 6)
        v_prom_periodo = st.slider("Promedio Materias/Período", 1.0, 8.0, 4.0, 0.5)
        v_per_reg = st.number_input("Períodos Regulares", 0, 5, 2)
        v_niv_max = st.selectbox("Nivel Máximo", [1, 2, 3, 4], index=1)
        v_niv_min = st.selectbox("Nivel Mínimo", [1, 2, 3, 4], index=0)
        v_esp = st.number_input("Registros Movilidad", 0, 6, 0)
        v_nota_cero = st.number_input("Materias con Nota 0", 0, 10, 0)
        v_asist_cero = st.number_input("Materias con Asistencia 0%", 0, 10, 0)
        v_max_vez = st.selectbox("Máx. Veces Cursó Materia", [1, 2, 3], index=0)
        v_mat_repetidas = st.number_input("Materias Repetidas", 0, 15, 0)
        v_prom_vez = st.slider("Promedio Veces/Materia", 1.0, 3.0, 1.0, 0.01)

    st.markdown("---")

    # Botón de predicción
    if st.button("🔍 **Predecir Riesgo de Deserción**", type="primary", use_container_width=True):

        # Construir diccionario con TODOS los features posibles
        datos = {
            'promedio_general': v_prom_gen, 'promedio_mediana': v_prom_gen,
            'promedio_min': v_prom_min, 'promedio_max': v_prom_max, 'promedio_std': v_prom_std,
            'asistencia_promedio': v_asist_prom, 'asistencia_mediana': v_asist_prom,
            'asistencia_min': v_asist_min, 'asistencia_max': 100, 'asistencia_std': v_asist_std,
            'materias_aprobadas': v_mat_apr, 'materias_reprobadas': v_mat_rep,
            'tasa_aprobacion': 1-v_tasa_rep, 'tasa_reprobacion': v_tasa_rep,
            'total_materias': v_total, 'max_vez_cursada': v_max_vez,
            'promedio_vez_cursada': v_prom_vez, 'materias_repetidas': v_mat_repetidas,
            'nivel_max': v_niv_max, 'nivel_min': v_niv_min,
            'num_periodos': v_per_reg+1, 'num_materias_distintas': v_mat_apr,
            'avance_niveles': v_niv_max-v_niv_min, 'num_periodos_regulares': v_per_reg,
            'materias_nota_cero': v_nota_cero, 'materias_asist_cero': v_asist_cero,
            'materias_nota_menor5': v_nota_cero+v_mat_rep,
            'materias_asist_menor50': v_asist_cero,
            'pct_nota_cero': v_nota_cero/v_total if v_total>0 else 0,
            'pct_asist_cero': v_asist_cero/v_total if v_total>0 else 0,
            'materias_ingles': 0, 'registros_especiales': v_esp,
            'materias_prom_periodo': v_prom_periodo, 'materias_max_periodo': v_max_periodo,
            'prom_primer': v_prom_gen, 'asist_primer': v_asist_prom,
            'prom_ultimo': v_prom_ult, 'asist_ultimo': v_asist_ult,
            'cambio_promedio': v_cambio_prom, 'cambio_asistencia': v_cambio_asist,
        }

        # Armar el vector con exactamente los features que espera el modelo
        X_input = pd.DataFrame([{f: datos.get(f, 0) for f in features}])
        X_scaled = pd.DataFrame(scaler.transform(X_input), columns=features)

        # Predicción
        pred = modelo.predict(X_scaled)[0]
        prob = modelo.predict_proba(X_scaled)[0]
        prob_des = prob[1] * 100

        # Mostrar resultado
        st.markdown("---")
        st.subheader("📋 Resultado")

        c1, c2 = st.columns([1, 2])
        with c1:
            if prob_des < 30:
                st.success(f"### 🟢 Riesgo BAJO\n### {prob_des:.1f}%")
            elif prob_des < 60:
                st.warning(f"### 🟡 Riesgo MEDIO\n### {prob_des:.1f}%")
            else:
                st.error(f"### 🔴 Riesgo ALTO\n### {prob_des:.1f}%")

            st.metric("Prob. Deserción", f"{prob_des:.1f}%")
            st.metric("Prob. Permanencia", f"{100-prob_des:.1f}%")

        with c2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_des,
                title={'text': "Probabilidad de Deserción (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#c0392b" if prob_des>60 else "#f39c12" if prob_des>30 else "#27ae60"},
                    'steps': [
                        {'range': [0, 30], 'color': '#d5f5e3'},
                        {'range': [30, 60], 'color': '#fdebd0'},
                        {'range': [60, 100], 'color': '#fadbd8'}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': prob_des}
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Recomendaciones según nivel de riesgo
        st.subheader("📝 Recomendaciones")
        if prob_des < 30:
            st.markdown("✅ El estudiante presenta un perfil académico saludable. Se recomienda **monitoreo regular**.")
        elif prob_des < 60:
            st.markdown("⚠️ Se detectan señales de riesgo moderado. Recomendaciones:")
            st.markdown("- Seguimiento académico más cercano\n- Tutoría personalizada\n- Revisión de carga académica")
        else:
            st.markdown("🚨 **Riesgo alto de deserción.** Acciones urgentes recomendadas:")
            st.markdown("- Intervención inmediata del tutor\n- Evaluación de situación personal\n- Plan de recuperación académica\n- Considerar reducción de carga")

# ================================================================================
# PÁGINA: EXPLORAR DATOS
# ================================================================================
elif pagina == "datos":
    st.title("📋 Exploración de Datos")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📄 Dataset Original", "⚙️ Dataset Procesado"])

    with tab1:
        st.subheader(f"Record Estudiantil ({len(df_orig):,} registros)")

        # Filtros interactivos
        c1, c2, c3 = st.columns(3)
        with c1: f_per = st.multiselect("Período", sorted(df_orig['PERIODO'].unique()))
        with c2: f_est = st.multiselect("Estado", df_orig['ESTADO'].unique())
        with c3: f_niv = st.multiselect("Nivel", sorted(df_orig['NIVEL'].unique()))

        df_f = df_orig.copy()
        if f_per: df_f = df_f[df_f['PERIODO'].isin(f_per)]
        if f_est: df_f = df_f[df_f['ESTADO'].isin(f_est)]
        if f_niv: df_f = df_f[df_f['NIVEL'].isin(f_niv)]

        st.caption(f"Mostrando {len(df_f):,} registros")
        st.dataframe(df_f, use_container_width=True, height=400)
        st.download_button("⬇️ Descargar CSV", df_f.to_csv(index=False), "datos_filtrados.csv")

    with tab2:
        st.subheader(f"Dataset Procesado ({len(df_mod)} estudiantes)")
        st.dataframe(df_mod, use_container_width=True, height=400)
        st.download_button("⬇️ Descargar CSV", df_mod.to_csv(index=False), "dataset_modelado.csv")

# ================================================================================
# FOOTER
# ================================================================================
st.sidebar.markdown("---")
st.sidebar.caption("Desarrollado con Streamlit\nMinería de Datos — CRISP-DM")
