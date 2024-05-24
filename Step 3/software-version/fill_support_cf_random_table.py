import yaml
import sys
import argparse
import pandas as pd
import psycopg2
import re
from sqlalchemy import create_engine

import log
logger = log.getlogger('random_crawling_support', level=log.DEBUG, filename='path/to/logs/random_crawling_support.log')

SPLIT_VERSION = re.compile(r'(.*?) ((\d+[. ]?)+)', re.S)

def load_config_yaml(conf_fname):
    with open(conf_fname) as s:
        config = yaml.safe_load(s)
    return config



def get_version(detected_component):
    match = SPLIT_VERSION.match(detected_component)
    if match:
        try:
            component = match.group(1).strip()
        except IndexError:
            component = None
        try:
            version = match.group(2).strip()
        except IndexError:
            version = None
    else:
        component, version = detected_component.strip(), None

    return {
        'what': component,
        'version': version
    }

def unfold_predictions(row):

    detected_components = row.iloc[23:]
    if detected_components[detected_components.notna()].empty:
        res = {
            'domain': row['domain'],
            'what': None,
            'version': None,
            'label': None
        }
        return [res]

    else:
        unfolded_components = detected_components[detected_components.notna()]\
                                .apply(lambda x: x.split(',') if ',' in x else x).explode()
        component_version = pd.json_normalize(unfolded_components.map(get_version))
        component_version['label'] = unfolded_components.index
        component_version['domain'] = row['domain']
        return component_version.to_dict(orient='records')


def persist(df):
    engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{dbname}'
                           .format(dbname=db_bindings['databases']['wptagent'], user=db_bindings['users']['user'],
                                   password=db_bindings['passwords']['password'],
                                   host=db_bindings['host'], port=db_bindings['port']), pool_pre_ping=True)

    try:
        df.to_sql('support_cfan_random_components', con=engine, if_exists='append', index=False, chunksize=500, method='multi')
    except Exception as e:
        logger.error(e)
        logger.info("Error inserting filehash {}.\n".format(filehash))
        sys.exit(-1)
    else:
        logger.debug(f'Insert of {df.shape[0]} records completed.')
    engine.dispose()


def first_run(db_bindings):
    with psycopg2.connect(dbname=db_bindings['databases']['wptagent'], user=db_bindings['users']['user'],
                          password=db_bindings['passwords']['password'], host=db_bindings['host']) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    select cf_analyses.domain, random_crawling.uri, random_crawling.page_data
                    from random_crawling
                    join cf_analyses using(domain_hash);""")
                scan_results = cur.fetchall()
            except psycopg2.errors.Error as e:
                logger.error(e)
                sys.exit(-1)

    df = pd.DataFrame(scan_results, columns=['domain', 'uri', 'page_data'])
    page_data = df.page_data.apply(pd.Series)
    merged = pd.concat([df, page_data], axis=1)
    components_list = merged.apply(unfold_predictions, axis=1)
    predictions_per_url = pd.DataFrame.from_dict(components_list.explode().to_list())

    predictions_per_url.drop_duplicates(inplace=True)

    persist(predictions_per_url)

def daily_update(db_bindings):
    with psycopg2.connect(dbname=db_bindings['databases']['wptagent'], user=db_bindings['users']['user'],
                          password=db_bindings['passwords']['password'], host=db_bindings['host']) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    select cf_analyses.domain, random_crawling.uri, random_crawling.page_data
                    from random_crawling
                    join cf_analyses using(domain_hash)
                    where timestamp > current_date - interval '1 day' + time '14:30';""")
                scan_results = cur.fetchall()
            except psycopg2.errors.Error as e:
                logger.error(e)
                sys.exit(-1)

    df = pd.DataFrame(scan_results, columns=['domain', 'uri' 'page_data'])
    if df.empty:
        logger.info('No update registered. Returning...')
        return
    page_data = df.page_data.apply(pd.Series)
    merged = pd.concat([df, page_data], axis=1)
    components_list = merged.apply(unfold_predictions, axis=1)
    predictions_per_domain = pd.DataFrame.from_dict(components_list.explode().to_list())
    predictions_per_domain.drop_duplicates(inplace=True) # shouldn't be necessary, but...

    persist(predictions_per_domain)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Producing intermediate data on the Compromised Frameworks analysis for Grafana.')
    parser.add_argument('--conf', dest='conf_fname', nargs='?', action='store', default="/path/to/conf/config.yaml",
                        help='Pipeline configuration file . (default: %(default)s)')
    args = parser.parse_args()

    config = load_config_yaml(args.conf_fname)
    db_bindings = config['global']['postgres']

    daily_update(db_bindings)
