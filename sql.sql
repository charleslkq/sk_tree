

create temporary function decrypt as 'com.yy.ent.hive.udf.DecryptUDF';
select
r5.noblelv,r5.uid,r5.pay_days,r5.pay_r,r5.pay,r5.all_avg_dr,r5.all_ndt,if(r6.uid is null,1,0)as is_lost
from 
(
select
r3.noblelv,r3.uid,r3.pay_days,r3.pay,r4.all_avg_dr,r4.all_ndt,datediff(r3.last_dt,'${month-2.start:y-m-d}')as pay_r
from 
(
select
r1.noblelv,r1.uid,r2.pay_days,r2.pay,concat(substr(r2.last_dt,0,4),'-',substr(r2.last_dt,5,2),'-',substr(r2.last_dt,7,2)) as last_dt
from 
(select noblelv,uid from yule.yule_nobleuser_identity_info_mt where mt='${month-2:y-m}')r1
join 
(
select 
  decrypt('entv587',accountid_e) as uid, 
  count(distinct dt)as pay_days,
  round(sum(if( moneytype = 10,decrypt('entv587' ,money_e)/100,decrypt('entv587' ,money_e)/1000 )),2) as pay,
  max(dt)as last_dt
from  yule.yule_product_paybill_original_day_e 
where  dt>='${month-2.start}' and dt<='${month-2.end}'
group by  accountid_e
)r2
on r1.uid=r2.uid
)r3
left outer join 
(
select uid,
round(sum(dr)/60/count(distinct dt),2) as all_avg_dr,
count(distinct dt) as all_ndt 
from 
(select dt,logtype,uid,sum(dr) as dr from 
 yule.yule_zyz_dr_original_day where dt>='${month-2.start:y-m-d}' and dt<='${month-2.end:y-m-d}'
and uid>0 and uid is not null and logtype in (0,2) group by dt,logtype,uid) aa group by uid
)r4
on r3.uid=r4.uid
)r5
left outer join 
(select  decrypt('entv587',accountid_e) as uid from  yule.yule_product_paybill_original_day_e where  dt>='${month-1.start}' and dt<='${month-1.end}' group by  accountid_e)r6
on r5.uid=r6.uid

 
 
 
 
 
 
 

