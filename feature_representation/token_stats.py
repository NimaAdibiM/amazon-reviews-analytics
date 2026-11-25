import bisect
import heapq
import math
from numbers import Integral

from .common import object_hash


UPPER_BOUNDS=[0]+[(1<<power)-1 for power in range(1,33)]
SAMPLE_LIMIT=128


def empty_histogram():
    return {'count':0,'sum':0,'min':None,'max':None,'counts':[0]*(len(UPPER_BOUNDS)+1)}


def add_value(histogram,value):
    histogram['count']+=1
    histogram['sum']+=value
    histogram['min']=value if histogram['min'] is None else min(histogram['min'],value)
    histogram['max']=value if histogram['max'] is None else max(histogram['max'],value)
    histogram['counts'][bisect.bisect_left(UPPER_BOUNDS,value)]+=1


def quantile_intervals(histogram):
    result={}
    for probability in (.5,.9,.95,.99):
        interval=None
        if histogram['count']:
            rank=math.ceil(probability*histogram['count'])
            total=0
            for index,count in enumerate(histogram['counts']):
                total+=count
                if total>=rank:
                    lower=0 if index==0 else UPPER_BOUNDS[index-1]+1
                    upper=UPPER_BOUNDS[index] if index<len(UPPER_BOUNDS) else histogram['max']
                    interval=[max(lower,histogram['min']),min(upper,histogram['max'])]
                    break
        result[str(probability)]=interval
    return result


class TokenSummary:
    def __init__(self,seed,namespace):
        self.seed=seed
        self.namespace=namespace
        self.lengths=empty_histogram()
        self.dropped=empty_histogram()
        self.truncated=0
        self.sample=[]

    def _sample(self,key,length,dropped):
        item=(-int(key,16),key,length,dropped)
        if len(self.sample)<SAMPLE_LIMIT:
            heapq.heappush(self.sample,item)
        elif item>self.sample[0]:
            heapq.heapreplace(self.sample,item)

    def add(self,identity,length,dropped):
        if any(isinstance(v,bool) or not isinstance(v,Integral) for v in (length,dropped)):
            raise ValueError('INVALID_TOKEN_COUNT')
        length,dropped=int(length),int(dropped)
        if length<0 or dropped<0 or dropped>length:
            raise ValueError('INVALID_TOKEN_COUNT')
        add_value(self.lengths,length)
        add_value(self.dropped,dropped)
        self.truncated+=int(dropped>0)
        self._sample(object_hash([self.seed,self.namespace,identity]),length,dropped)

    def merge(self,value):
        if value['schema']!='token-summary-1' or value['seed']!=self.seed or value['namespace']!=self.namespace or value['upper_bounds']!=UPPER_BOUNDS or value['sample_limit']!=SAMPLE_LIMIT:
            raise ValueError('TOKEN_SUMMARY_CONTRACT_MISMATCH')
        n = value['lengths']['count']
        for field in ('lengths','dropped'):
            h = value[field]
            if (len(h['counts']) != len(UPPER_BOUNDS)+1 or h['count']!=n
                    or any(type(x) is not int or x<0 for x in h['counts'])
                    or sum(h['counts']) != n or h['sum']<0
                    or (n and not 0<=h['min']<=h['max'])
                    or (not n and (h['min'] is not None or h['max'] is not None or h['sum']!=0))):
                raise ValueError('INVALID_TOKEN_HISTOGRAM')
        if not 0<=value['truncated_reviews']<=n or len(value['sample'])!=min(n,SAMPLE_LIMIT):
            raise ValueError('INVALID_TOKEN_SUMMARY_COUNTS')
        for item in value['sample']:
            key = item['key_hash']
            if (not isinstance(key,str) or len(key)!=64 or any(ch not in '0123456789abcdef' for ch in key)
                    or not 0<=item['dropped']<=item['length']):
                raise ValueError('INVALID_TOKEN_SAMPLE')
        for target,key in ((self.lengths,'lengths'),(self.dropped,'dropped')):
            source=value[key]
            target['count']+=source['count']
            target['sum']+=source['sum']
            target['counts']=[a+b for a,b in zip(target['counts'],source['counts'])]
            if source['count']:
                target['min']=source['min'] if target['min'] is None else min(target['min'],source['min'])
                target['max']=source['max'] if target['max'] is None else max(target['max'],source['max'])
        self.truncated+=value['truncated_reviews']
        for item in value['sample']:
            self._sample(item['key_hash'],item['length'],item['dropped'])

    def export(self):
        return {'schema':'token-summary-1','seed':self.seed,'namespace':self.namespace,
                'upper_bounds':UPPER_BOUNDS.copy(),'sample_limit':SAMPLE_LIMIT,
                'lengths':{**self.lengths,'counts':self.lengths['counts'].copy(),'quantile_intervals':quantile_intervals(self.lengths)},
                'dropped':{**self.dropped,'counts':self.dropped['counts'].copy(),'quantile_intervals':quantile_intervals(self.dropped)},
                'truncated_reviews':self.truncated,
                'sample':[{'key_hash':key,'length':length,'dropped':dropped} for _,key,length,dropped in sorted(self.sample,reverse=True)]}
