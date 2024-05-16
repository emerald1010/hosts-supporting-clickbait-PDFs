import argparse
import sys
import os
import psycopg2
from sqlalchemy import create_engine
import yaml
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
from numpy import nan
import re

from furl import furl
import time
from pwn import remote
import dns.resolver
from ipaddress import ip_address

from progress.bar import Bar
import log

# AS_REGEX = re.compile(r'AS([0-9]+) \w+', re.S)
# AS_REGEX = re.compile(r'.*?([0-9]+).*?', re.S)

AS_NUM_RE = re.compile(r'^\d+$')
IP_RE = re.compile(r'^\d+\.\d+\.\d+\.\d+$')
BGP_RE = re.compile(r'^\d+\.\d+\.\d+\.\d+(/\d{2})$')
CC_RE = re.compile(r'[A-Z]{2}')
REGISTRY_RE = re.compile((r'[a-z]+'))



def load_config_yaml(conf_fname):
    with open(conf_fname) as s:
        config = yaml.safe_load(s)
    return config

def parse_dns_records(dns_records):
    results = []

    if not dns_records or not isinstance(dns_records, list):
        return [defaultdict(lambda: None)]

    for record in dns_records:
        tmp_res = defaultdict(lambda: None)
        tmp_res['analysis_timestamp'] = datetime.now()
        if isinstance(record, tuple):
            assert( len(record) == 4 )
            tmp_res['domain'] = record[1]
            tmp_res['netloc'] = record[2]
            tmp_res[record[3]] = [str(x) for x in record[0]] # key of this is the RTYPE (A, AAAA, CNAME, ...) and value is the DNS answer
        else:
            if isinstance (record, dns.resolver.NoAnswer):
                split_1 = str(record)
            elif isinstance (record,  dns.resolver.NXDOMAIN):
                tmp_res['error'] = 'DNS query name does not exist.'
                split_1 = str(record)

            elif isinstance (record, dns.exception.Timeout):
                tmp_res['error'] = 'The DNS operation timed out.'
                split_1 = str(record)

            elif isinstance (record, dns.resolver.NoNameservers):
                tmp_res['error'] = 'Server 8.8.8.8 UDP port 53 answered SERVFAIL.'
                split_1 = str(record).split(':')[0]

            else:
                __LOGGER__.warning(record)
                split_1 = ''
                tmp_res['error'] = 'Error while processing the DNS record.'

            splits = split_1.split(' ')
            requested_domain = splits[-3][:-1] if splits[-3].endswith('.') else splits[-3]
            tmp_res['domain'] = requested_domain
            rtype = splits[-1]
            tmp_res[rtype] = nan
        results.append(tmp_res)


    df = pd.DataFrame(results, columns=['domain', 'netloc', 'analysis_timestamp', 'A', 'NS', 'CNAME', 'SOA', 'MX', 'TXT',
                                  'AAAA', 'CERT', 'A6', 'DNAME', 'CAA', 'error'])

    return df.groupby('domain').last().copy()

def parse_whois(line):
    res = defaultdict(lambda: None)

    info = line.split("|")  # [-7:]

    for split in info:
        ip = re.match(IP_RE, split.strip())
        if ip:
            res['IP'] = ip.group(0)
            continue
        bgp = re.match(BGP_RE, split.strip())
        if bgp:
            res['bgp_prefix'] = bgp.group(0)
            continue
        cc = re.match(CC_RE, split.strip())
        if cc:
            res['CC'] = cc.group(0)
            continue
        registry = re.match(REGISTRY_RE, split.strip())
        if registry:
            res['registry'] = registry.group(0)
            continue
        try:
            allocated = datetime.strptime(split.strip(), '%Y-%m-%d').date()
            res['allocated'] = allocated
            continue
        except ValueError:
            pass
        as_number = re.match(AS_NUM_RE, split.strip())
        if as_number:
            res['autonomous_sys'] = as_number.group(0)
            continue

    try:
        res['as_name'] = info[-1].strip()
    except IndexError as e:
        __LOGGER__.warning(line)
        __LOGGER__.exception(e)
    res['analysis_timestamp'] = datetime.now()

    return res

def whois(ip_list):
    # command = "whois -h whois.cymru.com \"-v {}\"".format(ip)
    # proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # output, error = proc.communicate()
    # if error:
    #     print('whois error')
    #     print(error.decode('utf-8'))
    #
    # dec_out = output.decode('utf-8')


    conn = None
    query = 'begin\nverbose'
    for ip in ip_list:
        query += f'\n{ip}'
    query += '\nend'
    lines = []
    try:
        conn = remote('whois.cymru.com', 43)
        time.sleep(2)
        conn.send(query)
        conn.recvline()

        c = True
        while c:
            try:
                # print('inside')
                lines += conn.recv().decode('utf-8').split('\n')[:-1]
            except:
                c = False
        # print('outside')
    except Exception as e:
        __LOGGER__.error(e)
    # r = r.decode('utf-8').split('\n')[:-1]
    conn.close()

    records = []
    for whois_lookup in lines:
        records.append(parse_whois(whois_lookup))

    return records

def get_dns_info(domain):
    """
    :param phish: The Phish object which contains the link and all the information about the current page
    :return: The dns_ns_rrset response
    """
    """
    Get all the records associated to domain parameter.
    :param domain:
    :return:
    """

    properties = ['A', 'NS', 'CNAME', 'SOA', 'MX', 'TXT', 'AAAA', 'CERT', 'A6', 'DNAME', 'CAA']
    result = defaultdict(list)

    for prop in properties:
        try:
            res = dns.resolver.Resolver(configure=False)
            res.nameservers = ['8.8.8.8']
            answers = res.resolve(domain, prop)
            for rdata in answers:
                result[prop].append(rdata.to_text())
        except dns.resolver.NoAnswer:
            pass
        except dns.resolver.NXDOMAIN:
            result['error'] = 'DNS query name does not exist.'
        # except dns.resolver.LifetimeTimeout: # server has version 2.1.0 and this does not exist
        except dns.exception.Timeout:
            # result['error'] = 'resolution lifetime expired.'
            result['error'] = 'The DNS operation timed out.'
        except dns.resolver.NoNameservers:
            result['error'] = 'Server 8.8.8.8 UDP port 53 answered SERVFAIL.'
        except dns.name.LabelTooLong as e2:
            __LOGGER__.exception(e2)
            print(domain)
            print(prop)
            result['error'] = 'A DNS label is > 63 octets long.'
        except Exception as e:
            __LOGGER__.exception(e)

    return result


def remove_redundancy(dns_records):
    with psycopg2.connect(dbname=db_bindings['databases']['pipeline'], user=db_bindings['users']['ginopino'], password=db_bindings['passwords']['ginopino'], host=db_bindings['host']) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT domain, netloc, analysis_timestamp, a, ns, cname, soa, mx, txt, aaaa, cert, a6, dname, caa, error
                    FROM dns_records
                    WHERE analysis_timestamp < %s; 
                """, ( datetime.today().strftime("%Y-%m-%d"), )) # WHERE is useless, records are not yet in DB, still...
                res = cur.fetchall()
            except psycopg2.errors.Error as e:
                __LOGGER__.error(e)
                res = []
    df_all = pd.DataFrame(res, columns=['domain', 'netloc', 'analysis_timestamp', 'A', 'NS', 'CNAME', 'SOA', 'MX', 'TXT',
                                  'AAAA', 'CERT', 'A6', 'DNAME', 'CAA', 'error'])
    previous_domain_records = df_all.sort_values('analysis_timestamp', ascending=False).drop_duplicates('domain', keep='first')

    merged = previous_domain_records.merge(dns_records, on='domain', how='right', indicator=True,
                                         suffixes=['_old', '_new'])
    to_insert = merged[merged._merge == 'right_only'][['domain', 'netloc_new', 'analysis_timestamp_new', 'A_new', 'NS_new', 'CNAME_new',
                                            'SOA_new', 'MX_new', 'TXT_new', 'AAAA_new', 'CERT_new', 'A6_new', 'DNAME_new',
                                            'CAA_new', 'error_new']] # these have no prior record. We are going to insert them anyway and no further processing is needed

    others = merged[merged._merge=='both']
    for i, grp in others.groupby('domain'):
        if i in to_insert.domain.unique():
            __LOGGER__.error('{} in to_insert already. Skipping. THIS SHOULD NOT HAPPEN...'.format(i))
            continue
        try:
            tmp = grp.A_new.transform(lambda x: set(x)).compare(grp.A_old.transform(lambda x: set(x)))
            if tmp.empty:
                # __LOGGER__.debug('DNS records for domain {} are equals. No insert needed.'.format(i))
                continue
            else:
                new = pd.concat([to_insert, grp[['domain', 'netloc_new', 'analysis_timestamp_new', 'A_new', 'NS_new', 'CNAME_new',
                                            'SOA_new', 'MX_new', 'TXT_new', 'AAAA_new', 'CERT_new', 'A6_new', 'DNAME_new',
                                            'CAA_new', 'error_new']]])
                to_insert = new.copy(deep=True)
        except (AttributeError, TypeError) as e:
            if grp.A_new.isna().all() and grp.A_old.isna().all():
                # __LOGGER__.debug('DNS records for domain {} are equals (None). No insert needed.'.format(i))
                continue
            elif not (grp.A_new.isna().all()) or not (grp.A_old.isna()).all():
                __LOGGER__.warning(e)
                new = pd.concat([to_insert, grp[['domain', 'netloc_new', 'analysis_timestamp_new', 'A_new', 'NS_new', 'CNAME_new',
                                            'SOA_new', 'MX_new', 'TXT_new', 'AAAA_new', 'CERT_new', 'A6_new', 'DNAME_new',
                                            'CAA_new', 'error_new']]])
                to_insert = new.copy(deep=True)
        except Exception as e2:
            __LOGGER__.exception(e2)
            __LOGGER__.info(i)
            __LOGGER__.info(grp)
            break

    to_insert.rename(mapper=lambda x: x.replace('_new', ''), axis=1, inplace=True) # go back to DB columns names
    return to_insert


def process_slice(slice):
    all_dns_records = []
    all_whois_records = []
    spurious_ips = []

    __LOGGER__.debug('Retrieving DNS records...')
    with Bar('hosts', max=step) as bar:
        for _, furl_domain, netloc in slice:
            analysis_timestamp = datetime.now()

            try:
                ip_address(furl_domain)
                spurious_ips.append(furl_domain)
                continue
            except ValueError:
                pass

            dns_record = get_dns_info(furl_domain)
            dns_record.update({
                'analysis_timestamp': analysis_timestamp,
                'domain': furl_domain,
                'netloc': netloc
            })
            all_dns_records.append(dns_record)
            bar.next()

    DNSes = pd.DataFrame(all_dns_records,
                         columns=['domain', 'netloc', 'analysis_timestamp', 'A', 'NS', 'CNAME', 'SOA', 'MX', 'TXT',
                                  'AAAA', 'CERT', 'A6', 'DNAME', 'CAA', 'error'])
    updated_or_new_DNSes = remove_redundancy(DNSes)
    updated_or_new_DNSes.rename(columns={
        'A': 'a',
        'NS': 'ns',
        'CNAME': 'cname',
        'SOA': 'soa',
        'MX': 'mx',
        'TXT': 'txt',
        'AAAA': 'aaaa',
        'CERT': 'cert',
        'A6': 'a6',
        'DNAME': 'dname',
        'CAA': 'caa'
    }, inplace=True) # names are lowercase in DB and mismatch causes everything to fail...
    __LOGGER__.debug('Inserting current chunk in `dns_records`.')

    engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{dbname}'
                           .format(dbname=db_bindings['databases']['pipeline'], user=db_bindings['users']['ginopino'], password=db_bindings['passwords']['ginopino'],
                                   host=db_bindings['host'], port=db_bindings['port']), pool_pre_ping=True)
    try:
        updated_or_new_DNSes.to_sql('dns_records', con=engine, if_exists='append', index=False, chunksize=1000, method='multi')
    except Exception as e:
        __LOGGER__.exception(e)

    As = updated_or_new_DNSes.explode('a')
    ips = As[As.a.notna()].a.unique()
    spurious_ips.extend(list(ips))
    __LOGGER__.debug('Fetching WHOIS data for {} IPs...'.format(len(spurious_ips)))
    all_whois_records.extend(whois(spurious_ips))

    WHOISs = pd.DataFrame(all_whois_records,
                          columns=['autonomous_sys', 'IP', 'bgp_prefix', 'CC', 'registry', 'allocated', 'as_name',
                                   'analysis_timestamp'])
    WHOISs.rename(columns={
        'IP': 'ip',
        'CC': 'cc'
    }, inplace=True)
    __LOGGER__.debug("Inserting current chunk in `whois_records`.")
    try:
        WHOISs.to_sql('whois_records', con=engine, if_exists='append', index=False, chunksize=1000, method='multi')
    except Exception as e:
        __LOGGER__.exception(e)

    engine.dispose()
    return


'''
NOTICE:
furl and urlparse have different behavior when the encoding of the URL changes, e.g.
'https://xn--72ca1bzcdf9cg5df4n5a8cei.com/userfiles/files/71910520548.pdf',
'http://xn--spreewaldpension-lbben-9lc.de/meineBilderAlbertGrundschule/file/dasawal.pdf', i.e.

                                                   url                             netloc                 furl_domain
139  https://xn--72ca1bzcdf9cg5df4n5a8cei.com/userf...   xn--72ca1bzcdf9cg5df4n5a8cei.com         ไม้ด่างแห่งสยาม.com
330  http://xn--spreewaldpension-lbben-9lc.de/meine...  xn--spreewaldpension-lbben-9lc.de  spreewaldpension-lübben.de

I'll use furl, but keep a reference to netloc in the table

'''





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get DNS records and WHOIS information for yesterday\'s URLs.')
    parser.add_argument('--conf', dest='conf_fname', nargs='?', action='store', default="/path/to/conf/config.yaml",
                        help='Pipeline configuration file . (default: %(default)s)')
    args = parser.parse_args()

    config = load_config_yaml(args.conf_fname)
    db_bindings = config['global']['postgres']

    log_file_path = os.path.join(config['global']['file_storage'], 'logs', 'host_info_gathering.log')
    global __LOGGER__; __LOGGER__ = log.getlogger(component='host_info_extraction', level=log.INFO, filename=log_file_path)

    import_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    with psycopg2.connect(dbname=db_bindings['databases']['pipeline'], user=db_bindings['users']['ginopino'], password=db_bindings['passwords']['ginopino'],
                          host=db_bindings['host']) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    select uri, netloc
                    from all_urls 
                    join imported_samples using(filehash)
                    where   imported_samples.provider <> 'FromUrl'
                            and all_urls.is_pdf = True
                            AND all_urls.has_http_scheme = True
                            AND all_urls.is_email = FALSE
                            AND all_urls.is_empty = FALSE
                            AND all_urls.is_relative_url = FALSE
                            AND all_urls.is_local = FALSE
                            AND all_urls.has_invalid_char = FALSE
                            AND all_urls.has_valid_tld = True
                            AND imported_samples.upload_date = %s;
                """, (import_date,))
                res = cur.fetchall()
                __LOGGER__.info(f"SELECTed {len(res)} records.")
            except psycopg2.errors.Error as e:
                __LOGGER__.error(e)
                res = []

    df = pd.DataFrame(res, columns=['url', 'netloc'])
    if df.empty:
        __LOGGER__.info('DF is empty. Returning...')
        sys.exit(0)
    df['furl_domain'] = df.apply(lambda x: furl(x['url']).host, axis=1)
    df.drop_duplicates('furl_domain', inplace=True)

    step = 5000
    db_records = df.apply(tuple, axis=1).tolist()
    times = int(len(db_records) / step)
    remainder = len(db_records) % step

    for i in range(times):
        process_slice(db_records[i * step: (i + 1) * step])
    if remainder != 0:
        process_slice(db_records[-remainder:])

