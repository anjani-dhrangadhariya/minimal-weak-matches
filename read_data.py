import pandas as pd
import ast
import json



def get_text(words, offsets):

    S = []
    for w, o in zip( words, offsets ):
        s = ''
        for i, term in zip(o, w):
            if len(s) == i:
                s += str(term)
            elif len(s) < i:
                s += (' ' * (i - len(s))) + str(term)
            else:
                raise Exception('text offset error')
        
        S.append( s )

    return S

# read a source file into a df
def get_source_df(fpath):

    tokens = []
    text = []
    pos = []
    pos_fine = []
    lemma = []

    counter = 0
    with open(fpath, 'r') as rf:
        for eachSource in rf:
            source_i = ast.literal_eval( eachSource )
            for k,v in source_i.items():
                if 'tokens' in v:
                    tokens.append( v['tokens'] )
                if 'text' in v:
                    text.append( v['text'] )
                if 'pos' in v:
                    pos.append( v['pos'] )
                if 'pos_fine' in v:
                    pos_fine.append( v['pos_fine'] )
                if 'lemma' in v:
                    lemma.append( v['lemma'] )

            # counter = counter + 1
            # if counter == 10:
            #     break

    df = pd.DataFrame( {'tokens': tokens, 'text': text, 'pos': pos, 'pos_fine': pos_fine, 'lemma': lemma })

    return df


def get_target_df(train_dir, write_to_file):

    pmid = []
    text = []
    tokens = []
    pos = []
    char_offsets = []
    p = []
    i = []
    o = []

    with open(f'{train_dir}', 'r') as rf:
        data = json.load(rf)
        for k,v in data.items():
            pmid.append( k )
            tokens.append( [x.strip() for x in v['tokens'] ] )
            pos.append( v['pos'] )
            char_offsets.append( v['abs_char_offsets'] )
            
            if 'participants' in v:
                vp = v['participants']
                p.append( vp )
            else:
                p.append( [ '0' ] * len( v['tokens'] ) )

            if 'interventions' in v:
                vi = v['interventions']
                i.append( vi )
            else:
                i.append( [ '0' ] * len( v['tokens'] ) )

            if 'outcomes' in v:
                vo = v['outcomes']
                o.append( vo  )
            else:
                o.append( [ '0' ] * len( v['tokens'] ) )
    

    df_data = pd.DataFrame( {'pmid': pmid, 'tokens': tokens, 'pos': pos, 'offsets': char_offsets, 'p': p, 'i': i, 'o': o } )
    
    text = get_text(df_data['tokens'], df_data['offsets'])
    df_data['text'] = text

    df_data_token_flatten = [item for sublist in list(df_data['tokens']) for item in sublist]
    df_data_pos_flatten = [item for sublist in list(df_data['pos']) for item in sublist]
    df_data_offset_flatten = [item for sublist in list(df_data['offsets']) for item in sublist]

    df_data_p_labels_flatten = [item for sublist in list(df_data['p']) for item in sublist]
    df_data_p_labels_flatten = list(map(int, df_data_p_labels_flatten))

    df_data_i_labels_flatten = [item for sublist in list(df_data['i']) for item in sublist]
    df_data_i_labels_flatten = list(map(int, df_data_i_labels_flatten))

    df_data_o_labels_flatten = [item for sublist in list(df_data['o']) for item in sublist]
    df_data_o_labels_flatten = list(map(int, df_data_o_labels_flatten))


    return df_data