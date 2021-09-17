import base64
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances
import streamlit as st

df = pd.read_excel('jojo_stand.xlsx')
df.head()

def letter_to_number(x):
    if x == 'A':
        return 5
    elif x == 'B':
        return 4
    elif x == 'C':
        return 3
    elif x == 'D':
        return 2
    if x == 'E':
        return 1
    else:
        return np.nan

df_atributes = df.iloc[:,2:].applymap(lambda x: letter_to_number(x))
df_nome = pd.DataFrame(df['stand_name'])

df_novo = df_nome.reset_index().merge(df_atributes.reset_index()).drop(columns=['index'])



def render_mpl_table(data, col_width=5.0, row_height=0.625, font_size=18,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) * np.array([col_width, row_height]))
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w',fontsize=22, fontfamily='serif')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax


def calcular_similaridade(stand, n):
    try:
        stand_select = df_novo[df_novo['stand_name'] == stand].iloc[:,1:].values
        stand_dist = []
        for other_stand in df_novo['stand_name'].tolist():
            other_stand = df_novo[df_novo['stand_name'] == other_stand].iloc[:,1:].values
            stand_dist.append(nan_euclidean_distances(stand_select, other_stand))
        stand_dist_df = pd.DataFrame(zip(df_novo['stand_name'].tolist(), stand_dist), columns=['stand_name', 'stand_distance'])
        df_final = df_novo.merge(stand_dist_df, how = 'left', on = 'stand_name')
        df_final['stand_distance'] = df_final['stand_distance'].apply(lambda x : str(x).replace('[[','').split(']')[0])
        df_final = df_final.sort_values(by = 'stand_distance', ascending=True).dropna().reset_index().drop(columns=['index'])
        df_final['stand_name'] = df_final['stand_name'].drop_duplicates(keep='first')
        df_final['stand_distance'] = df_final['stand_distance'].apply(lambda x: round(float(x.split(' ')[0]), 2))
        df_final = df_final.drop(df_final[df_final['stand_distance'] == 0.00].index)
        print(f'Stands similares a: {stand}')
        return st.table(df_final.head(n))
    except:
        return 'Stand não encontrado. Por favor, verifique se o nome foi digitado corretamente'


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
      background-image: url("data:image/png;base64,%s");
      background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('wallpaper.png')

st.markdown("# Recomendador de stands - Jojo's Bizarres Adventures")
st.write('### Um sistema de recomendação de personagens baseado nos atributos')
st.write('')
st.write('')
st.write('')
stand_escolhido = st.selectbox('Selecione um stand:', df['stand_name'])


calcular_similaridade(stand_escolhido, 10)