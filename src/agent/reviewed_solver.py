#!/usr/bin/env python3
"""Execute semantic plans using only documented simulator tools and history."""
from __future__ import annotations
import argparse
import json
import pickle
import sys
import unicodedata
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'public_test'))
from zaic_tools import ToolEnv


def fold(value):
    return ''.join(c for c in unicodedata.normalize('NFD', str(value).lower().replace('đ','d')) if not unicodedata.combining(c))


class Solver:
    def __init__(self,item,session,save):
        self.item,self.env,self.save=item,session,save
        self.now=datetime.fromisoformat(item['context']['current_time'])
        self.profile=item['context']['user_profile']
        self.history=[]
        for turn in item['history']:
            self.history.extend(zip(turn.get('tool_calls',[]),turn.get('tool_results',[])))
        self.history.extend(({'name':x['name'],'arguments':x['arguments']},x['result']) for x in session.log)

    def call(self,_function,**args):
        args={k:v for k,v in args.items() if v is not None}
        check=self.env.validate(_function,args)
        if not check['ok']:
            if 'không thuộc app đã cài' in check.get('error',''):
                return {'ok':False,'error':check['error']}
            raise ValueError(f"{self.item['id']}: {_function} {args}: {check}")
        result=self.env.call(_function,args)
        self.history.append(({'name':_function,'arguments':args},result))
        self.save()
        return result

    def choose(self,values,query=None,filters=None):
        values=[v for v in values if all(str(v.get(k,''))==str(x) for k,x in (filters or {}).items())]
        if not values:return None
        if not query:return values[0] if len(values)==1 else None
        q=fold(query)
        def score(v):
            name=fold(v.get('name') or v.get('title') or v.get('lot_name') or v.get('operator') or v.get('flight_no') or v.get('train_no') or v.get('id'))
            if q==name:return 10
            if q in name:return 8-len(name)/10000
            words=q.split()
            return sum(w in name for w in words)/len(words)+SequenceMatcher(None,q,name).ratio()/10
        ranked=sorted(values,key=score,reverse=True)
        if score(ranked[0])<0.45:return None
        if len(ranked)==1 or score(ranked[0])>score(ranked[1])+0.001:
            return ranked[0]
        return None

    def lookup(self,tool,collection,query=None,args=None,filters=None):
        for call,result in reversed(self.history):
            if call['name']==tool:
                found=self.choose(result.get(collection,[]),query,filters)
                if found:return found
        result=self.call(tool,**(args or {}))
        return self.choose(result.get(collection,[]),query,filters)

    def shop(self,query,quantity=1,app='grocery',checkout=True,payment=None,delivery_time=None,choose_query=None):
        product=self.lookup(app+'.search','products',choose_query or query,{'query':query})
        if not product:return self.ask(app+'.add_to_cart','product_id')
        self.call(app+'.add_to_cart',product_id=product['id'],quantity=quantity)
        if checkout:
            kwargs={}
            if app=='ecommerce' and payment:kwargs['payment']=payment
            if app=='grocery' and delivery_time:kwargs['delivery_time']=self.when(delivery_time)
            self.call(app+'.checkout',**kwargs)

    def return_order(self,query=None,reason='damaged',cancel=False):
        values=[]
        known=self.known('ecommerce.order_status')
        if known:values=known.get('orders',[])
        def select(values):
            candidates=[]
            for v in values:
                text=json.dumps(v,ensure_ascii=False)
                if query is None or fold(query) in fold(text):candidates.append(v)
            return candidates[0] if len(candidates)==1 else None
        value=select(values)
        if not value:value=select(self.call('ecommerce.order_status').get('orders',[]))
        name='ecommerce.cancel_order' if cancel else 'ecommerce.request_return'
        if not value:return self.ask(name,'order_id')
        self.call(name,order_id=value['id'],**({} if cancel else {'reason':reason}))

    def ticket(self,query,quantity=1,date=None,calendar=False,calendar_start=None,title=None,ride_before=None,remind=False,max_price=None,only_available=False):
        e=self.lookup('event_tickets.search','events',query,{'query':query},{'date':date} if date else None)
        if not e:return self.ask('event_tickets.book','event_id')
        if max_price is not None and e.get('price',0)>=max_price:return
        if only_available and e.get('available',0)<quantity:return
        self.call('event_tickets.book',event_id=e['id'],quantity=quantity)
        if calendar:
            start=calendar_start or (e['date']+'T'+e['time']+':00+07:00')
            self.call('calendar.create_event',title=title or 'đi xem '+e['name'],start=start,location=e.get('venue'))
        if ride_before is not None:
            pickup=datetime.fromisoformat(e['date']+'T'+e['time']+':00+07:00')-timedelta(minutes=ride_before)
            self.call('ride_hailing.book',pickup=self.profile['home_address'],dropoff=e['venue'],vehicle='car4',schedule_time=pickup.isoformat())
            if remind:self.reminder(pickup.isoformat(),'ra khỏi nhà đi xem '+e['name'])

    def train_trip(self,destination,date,code=None,seats=1,origin='Hà Nội',parking=None,ride=None,calendar=False,arrive_before=None):
        prior=self.known('train_bus_tickets.search')
        trips=prior.get('trips',[]) if prior else []
        if not any(t.get('date')==date for t in trips):
            trips=self.call('train_bus_tickets.search',origin=origin,destination=destination,date=date,type='train').get('trips',[])
        trips=[t for t in trips if t.get('date')==date]
        if arrive_before:trips=[t for t in trips if t.get('arrive','')<arrive_before]
        t=self.choose(trips,code) if code else (trips[0] if len(trips)==1 else None)
        if not t:return self.ask('train_bus_tickets.book','trip_id')
        self.call('train_bus_tickets.book',trip_id=t['id'],seats=seats)
        if parking:self.parking(**parking)
        if ride:self.call('ride_hailing.book',pickup=self.profile['home_address'],dropoff=ride.get('dropoff','Ga Hà Nội'),vehicle=ride.get('vehicle','car4'),schedule_time=self.when(ride['time']) if ride.get('time') else None)
        if calendar:
            self.call('calendar.create_event',title='đi tàu '+str(code or destination),start=date+'T'+t['depart']+':00+07:00',end=date+'T'+t['arrive']+':00+07:00')

    def parking(self,near,time,hours,query=None):
        if near=='$work':near=self.profile['work_address']
        if near=='$home':near=self.profile['home_address']
        lot=self.lookup('parking.find','lots',query or near,{'near':near})
        if not lot:return self.ask('parking.reserve','lot_id')
        self.call('parking.reserve',lot_id=lot['id'],from_time=self.when(time),hours=hours)

    def recipe(self,query,servings=None,save=False,note=None,full=False):
        r=self.lookup('recipes.search','recipes',query,{'query':query,**({'servings':servings} if servings else {})})
        if not r:return
        if full:self.call('recipes.get',recipe_id=r['id'])
        if save:self.call('recipes.save',recipe_id=r['id'])
        if note:self.call('notes.create',title=note,content=r['name'])

    def consume(self,query,quantity):
        item=self.lookup('pantry.list_items','items',query)
        if item:self.call('pantry.consume',item_id=item['id'],quantity=quantity)
        else:self.ask('pantry.consume','item_id')

    def translate(self,text,lang='en',note=None,to=None,subject=None,chat=None):
        result=self.call('translator.translate',text=text,to_lang=lang)
        translated=result.get('translation') or result.get('translated') or text
        if note:self.call('notes.create',title=note,content=translated)
        if to:self.call('email.create_draft',to=[to] if isinstance(to,str) else to,subject=subject or note or 'Thông báo',body=translated)
        if chat:self.chat(chat,translated)

    def email_contact(self,query,subject,body,relation=None):
        contacts=self.contacts(query)
        if relation:contacts=[c for c in contacts if fold(relation) in fold(c.get('relation','')+' '+c.get('name',''))]
        contacts=[c for c in contacts if c.get('email')]
        if len(contacts)!=1:return self.ask('email.create_draft','to')
        self.call('email.create_draft',to=[contacts[0]['email']],subject=subject,body=body)

    def call_last_contact(self):
        import re
        texts=[turn.get('text','') for turn in self.item['history'] if turn['role']=='assistant']
        for text in reversed(texts):
            hit=re.search(r'Số của (.+?) là (\d+)',text)
            if hit:
                name=hit[1]
                name=name.split(' (')[0]
                vals=self.contacts(name)
                vals=[c for c in vals if c.get('phone')==hit[2]]
                if len(vals)==1:return self.call('phone.call',contact_id=vals[0]['id'])
        self.ask('phone.call','contact_id')

    def cancel_index_event(self,index=2):
        result=self.read('calendar.list_events')
        # Display order is the order shown in the assistant's last list.
        events=result.get('events',[])
        if len(events)>=index:self.call('calendar.cancel_event',event_id=events[index-1]['id'])

    def temperature_fan(self,threshold=30,room='phòng khách'):
        result=self.read('weather.get_current')
        current=result.get('current',[])
        if any(x.get('temperature_c',0)>threshold for x in current):
            self.smart(fan={'room':room})

    def habit(self,query,date=None):
        h=self.lookup('habit_tracker.list_habits','habits',query)
        if h:self.call('habit_tracker.log',habit_id=h['id'],date=date or self.now.date().isoformat())
        else:self.ask('habit_tracker.log','habit_id')

    def music_play(self,query):
        t=self.lookup('music.list_tracks','tracks',query)
        if t:self.call('music.play',track_id=t['id'])

    def shopping_action(self,query,action='check'):
        item=self.lookup('shopping_list.get','items',query)
        if item:self.call('shopping_list.'+action,item_id=item['id'])

    def map_plan(self,query=None,near=None,destination=None,ride=None,route=False,transit=False,parking=None,remind=None):
        place=None
        if query:
            place=self.lookup('maps.search_place','places',query,{'query':query,**({'near':near} if near else {})})
            if not place:return
            destination=place.get('address') or place.get('name')
        if not destination:return
        if route:self.call('maps.get_route',origin=self.profile['home_address'],destination=destination)
        if transit:self.call('public_transit.find_routes',origin=self.profile['home_address'],destination=destination)
        if ride:self.call('ride_hailing.book',pickup=self.profile['home_address'],dropoff=destination,vehicle=ride)
        if parking:
            self.parking(near=place.get('name') if place else destination,**parking)
        if remind:
            self.reminder((self.now+timedelta(minutes=remind)).isoformat(),'đi tới '+destination)

    def recipe_pantry(self,query,servings=None,consume=None,shop=False,save=False,main_count=None):
        import math
        r=self.lookup('recipes.search','recipes',query,{'query':query,**({'servings':servings} if servings else {})})
        if not r:return
        pantry=self.read('pantry.list_items').get('items',[])
        for name,quantity in (consume or {}).items():
            p=self.choose(pantry,name)
            if p:
                self.call('pantry.consume',item_id=p['id'],quantity=quantity)
                p=dict(p);p['stock']=max(0,p.get('stock',p.get('quantity',0))-quantity)
                pantry=[p if x['id']==p['id'] else x for x in pantry]
        ingredients=r.get('ingredients',[])
        if main_count is not None:ingredients=ingredients[:main_count]
        for ing in ingredients:
            p=self.choose(pantry,ing.get('name'))
            available=p.get('stock',p.get('quantity',0)) if p else 0
            if p and fold(p.get('unit'))=='kg' and fold(ing.get('unit'))=='g':available*=1000
            already_used=sum(q for n,q in (consume or {}).items() if fold(n)==fold(ing.get('name')))
            needed=max(0,(ing.get('quantity') or 1)-available-already_used)
            if needed:
                if shop:self.buy_ingredient(ing['name'],needed,ing.get('unit'))
                else:self.call('shopping_list.add',name=ing['name'],quantity=max(1,math.ceil(needed)))
        if shop:self.call('grocery.checkout')
        if save:self.call('recipes.save',recipe_id=r['id'])

    def restock(self,query,consume,minimum,product=None):
        item=self.lookup('pantry.list_items','items',query)
        if not item:return
        self.call('pantry.consume',item_id=item['id'],quantity=consume)
        remain=max(0,item.get('stock',item.get('quantity',0))-consume)
        if remain<minimum:self.shop(product or query,minimum-remain)

    def buy_ingredient(self,query,quantity,unit):
        import re,math
        p=self.lookup('grocery.search','products',query,{'query':query})
        if not p:return self.ask('grocery.add_to_cart','product_id')
        count=quantity
        amount=re.search(r'(\d+(?:[.,]\d+)?)\s*(kg|ml|g|l)\b',fold(p.get('name','')))
        if amount:
            pack=float(amount[1].replace(',','.'));pack_unit=amount[2]
            if fold(unit)=='g' and pack_unit=='kg':pack*=1000;pack_unit='g'
            if fold(unit)=='ml' and pack_unit=='l':pack*=1000;pack_unit='ml'
            if fold(unit)==pack_unit:count=quantity/pack
        elif fold(unit) in ['g','ml'] and fold(p.get('unit')) in ['kg','l']:
            count=quantity/1000
        self.call('grocery.add_to_cart',product_id=p['id'],quantity=max(1,math.ceil(count)))

    def smart_conditional(self,room,brightness,power='on'):
        ds=self.call('smart_home.get_status',room=room).get('devices',[])
        for d in ds:
            if d.get('type')=='light' and d.get('state',{}).get('power')==power:
                self.call('smart_home.light_set',device_id=d['id'],brightness=brightness)

    def recipe_combo(self,query,forbidden='sữa',quantity=3):
        rs=self.read('recipes.search',query=query).get('recipes',[])
        r=self.choose(rs,query)
        if not r:return
        if fold(forbidden) in fold(json.dumps(r.get('ingredients',[]),ensure_ascii=False)):return
        self.shop('combo '+r['name'],quantity=quantity)

    def read_chat(self,query):
        chats=self.read('chat_ott.list_chats').get('chats',[])
        c=self.choose(chats,query)
        if c:return self.call('chat_ott.read_chat',chat_id=c['id'])

    def chat_email(self,chat_query,subject,body):
        result=self.read_chat(chat_query)
        if not result:return self.ask('email.create_draft','to')
        import re
        text=json.dumps(result,ensure_ascii=False)
        addresses=re.findall(r'[\w.+-]+@[\w.-]+\.[A-Za-z]+',text)
        if not addresses:
            contacts=self.contacts(chat_query)
            addresses=[c['email'] for c in contacts if c.get('email')]
        if len(set(addresses))==1:self.call('email.create_draft',to=[addresses[0]],subject=subject,body=body)
        else:self.ask('email.create_draft','to')

    def complete_task(self,query):
        t=self.lookup('tasks.list','tasks',query)
        if t:self.call('tasks.complete',task_id=t['id'])

    def transit_reminder(self,origin,destination,date,arrival,early=20):
        routes=self.call('public_transit.find_routes',origin=origin,destination=destination).get('routes',[])
        if not routes:return
        r=min(routes,key=lambda x:x.get('stops_count',999))
        self.call('public_transit.next_departures',stop=r.get('board_at') or origin,route=r.get('name'))
        base=datetime.fromisoformat(date+'T'+arrival+':00+07:00')-timedelta(minutes=r.get('duration_min',30)+early)
        self.reminder(base.isoformat(),'ra bến xe đi '+destination)

    def known(self,name,args=None):
        args=args or {}
        for call,result in reversed(self.history):
            if call['name']==name and all(call['arguments'].get(k)==v for k,v in args.items()):
                return result
        return None

    def read(self,name,**args):
        result=self.known(name,args)
        return result if result is not None else self.call(name,**args)

    def ask(self,tool,*params):
        check=self.env.validate(tool,{})
        if 'không thuộc app' in (check.get('error') or ''):return
        return self.call('sys.ask_missing_info',tool_name=tool,params=list(params))

    def when(self,value):
        if isinstance(value,str) and value.startswith('2026-'):
            return value
        # [day offset, hour, minute], evaluated in the sample's timezone.
        offset,hour,minute=value
        return (self.now+timedelta(days=offset)).replace(hour=hour,minute=minute,second=0,microsecond=0).isoformat()

    def reminder(self,time,content):
        return self.call('reminders.create',content=content,time=self.when(time))

    def wallet_bill(self,query,bill_type,relation=None):
        bills=self.read('bill_payment.list_bills').get('bills',[])
        b=next((x for x in bills if x.get('type')==bill_type and not x.get('paid')),None)
        if b:self.contact_action(query,'wallet',amount=b['amount'],relation=relation)

    def alarm_wallet(self,time,query):
        values=self.contacts(query)
        self.alarm(time)
        self.ask('ewallet.transfer',*(['contact_id','amount'] if len(values)!=1 else ['amount']))

    def chat(self,query,content):
        chats=self.read('chat_ott.list_chats').get('chats',[])
        values=[c for c in chats if fold(query) in fold(c.get('name',''))]
        if len(values)!=1:return self.ask('chat_ott.send_message','chat_id')
        self.call('chat_ott.send_message',chat_id=values[0]['id'],content=content)

    def pay_filtered(self,days=None,before=None,max_amount=None,exclude=None,source='bank'):
        bills=self.read('bill_payment.list_bills').get('bills',[])
        for b in bills:
            if b.get('paid') or b.get('type')==exclude:continue
            if max_amount is not None and b['amount']>=max_amount:continue
            due=b.get('due_date','')
            if before and due>=before:continue
            if days is not None and not (self.now.date().isoformat()<=due<=(self.now+timedelta(days=days)).date().isoformat()):continue
            self.call('bill_payment.pay',bill_id=b['id'],source=source)

    def alarm_event(self,title,offset=1,before=60):
        e=self.event(title,offset)
        if not e:return self.ask('alarm.create','time')
        self.alarm((datetime.fromisoformat(e['start'])-timedelta(minutes=before)).isoformat())

    def rain_alarm(self,minutes=30):
        fs=self.forecast()
        if not any(f.get('rain') for f in fs):return
        alarms=self.read('alarm.list').get('alarms',[])
        deleted={c['arguments']['alarm_id'] for c,r in self.history if c['name']=='alarm.delete'}
        alarms=[x for x in alarms if x['id'] not in deleted]
        if len(alarms)!=1:return self.ask('alarm.update','alarm_id')
        self.call('alarm.update',alarm_id=alarms[0]['id'],time=(datetime.fromisoformat(alarms[0]['time'])+timedelta(minutes=minutes)).isoformat())

    def alarm(self,time,label=None):
        return self.call('alarm.create',time=self.when(time),label=label)

    def cancel_event(self,title,offset=1):
        e=self.event(title,offset)
        if e:self.call('calendar.cancel_event',event_id=e['id'])

    def cancel_food(self,all_pending=True,query=None):
        orders=self.read('food_delivery.order_history').get('orders',[])
        for o in orders:
            if o.get('status') in ['placed','pending','preparing','accepted'] and (query is None or fold(query) in fold(json.dumps(o,ensure_ascii=False))):
                self.call('food_delivery.cancel_order',order_id=o['id'])

    def food(self,item,quantity=1,restaurant=None):
        rs=self.read('food_delivery.list_restaurants').get('restaurants',[])
        candidates=[self.choose(rs,restaurant)] if restaurant else rs
        for r in candidates:
            if not r:continue
            menu=self.call('food_delivery.get_menu',restaurant_id=r['id']).get('menu',[])
            values=[m for m in menu if fold(item) in fold(m.get('name',''))]
            if values:
                self.call('food_delivery.place_order',restaurant_id=r['id'],items=[{'item_id':values[0]['id'],'quantity':quantity}],address=self.profile['home_address'])
                return

    def end_parking(self,query=None):
        sessions=self.read('parking.list_sessions').get('sessions',[])
        values=[s for s in sessions if s.get('status') not in ['ended','cancelled']]
        s=self.choose(values,query) if query else (values[0] if len(values)==1 else None)
        if s:self.call('parking.end_session',session_id=s['id'])

    def cancel_train(self,destination=None):
        bs=self.read('train_bus_tickets.my_bookings').get('bookings',[])
        active=[b for b in bs if b.get('status')!='cancelled']
        trips={t['id']:t for c,r in self.history if c['name']=='train_bus_tickets.search' for t in r.get('trips',[])}
        for b in bs:
            if b.get('status')=='cancelled':continue
            if destination and len(active)!=1 and fold(destination) not in fold(json.dumps(trips.get(b.get('trip_id'),b),ensure_ascii=False)):continue
            self.call('train_bus_tickets.cancel',booking_id=b['id'])

    def bank_savings(self,amount,months):
        accounts=self.read('banking.get_balance').get('accounts',[])
        a=self.choose(accounts,'thanh toán')
        if a:self.call('banking.open_savings',from_account_id=a['id'],amount=amount,term_months=months)

    def stock_change(self,symbol,quantity=100):
        quote=self.read('stocks.get_quote',symbol=symbol).get('quote',{})
        change=quote.get('change_percent',quote.get('change_pct',quote.get('change',0)))
        if change:self.call('stocks.place_order',symbol=symbol,side='sell' if change>0 else 'buy',quantity=quantity)

    def stock_winners(self,sell=None,notify=None):
        portfolio=self.read('stocks.get_portfolio')
        values=portfolio.get('positions') or portfolio.get('portfolio') or portfolio.get('holdings') or []
        winners=[x for x in values if x.get('price',0)>x.get('avg_price',float('inf'))]
        if sell:
            for x in winners:self.call('stocks.place_order',symbol=x['symbol'],side='sell',quantity=min(sell,x.get('quantity',sell)))
        if notify:self.contact_action(notify,'sms',content='Các mã đang lời: '+', '.join(x['symbol'] for x in winners))

    def weather_park(self,near,time,hours=2,offset=1,query=None):
        fs=self.forecast(offset)
        if any(x.get('rain') for x in fs):self.parking(near,self.when(time),hours,query=query)

    def clear_bought(self):
        values=self.read('shopping_list.get').get('items',[])
        for x in values:
            if x.get('checked') or x.get('bought'):self.call('shopping_list.remove',item_id=x['id'])

    def recipe_reminder(self,query,threshold=60,long_time=None,short_time=None):
        r=self.lookup('recipes.search','recipes',query,{'query':query})
        if not r:return
        details=self.call('recipes.get',recipe_id=r['id']).get('recipe',r)
        duration=details.get('time_min',details.get('cook_time_min',details.get('duration_min',0)))
        self.reminder(long_time if duration>threshold else short_time,'bắt đầu nấu '+r['name'])

    def event_location_weather(self,title,offset,location):
        forecasts=self.forecast(offset)
        if any(x.get('rain') for x in forecasts):return
        e=self.event(title,offset)
        if e:self.call('calendar.update_event',event_id=e['id'],location=location)

    def transit_ride(self,destination,date,leave,dropoff,vehicle='car4',calendar=False):
        rs=self.call('public_transit.find_routes',origin=self.profile['home_address'],destination=destination).get('routes',[])
        if not rs:return
        r=min(rs,key=lambda x:x.get('stops_count',999))
        arrival=datetime.fromisoformat(date+'T'+leave+':00+07:00')+timedelta(minutes=r.get('duration_min',30))
        self.call('ride_hailing.book',pickup=r['alight_at'],dropoff=dropoff,vehicle=vehicle,schedule_time=arrival.isoformat())
        if calendar:self.call('calendar.create_event',title='Đi '+dropoff,start=arrival.isoformat())

    def transit_shift(self,title,offset,destination,leave,deadline):
        rs=self.call('public_transit.find_routes',origin=self.profile['home_address'],destination=destination).get('routes',[])
        if not rs:return
        r=min(rs,key=lambda x:x.get('stops_count',999))
        date=(self.now+timedelta(days=offset)).date().isoformat()
        arrival=datetime.fromisoformat(date+'T'+leave+':00+07:00')+timedelta(minutes=r.get('duration_min',30))
        if arrival.strftime('%H:%M')>=deadline:return
        e=self.event(title,offset)
        if e:self.call('calendar.update_event',event_id=e['id'],start=arrival.isoformat(),end=(arrival+timedelta(hours=1)).isoformat(),location=destination)

    def chat_cancel(self,query,title,offset):
        result=self.read_chat(query)
        text=fold(json.dumps(result,ensure_ascii=False))
        if 'huy' in text or 'nghi' in text:self.cancel_event(title,offset)

    def consume_all_restock(self,query,required):
        item=self.lookup('pantry.list_items','items',query)
        have=item.get('stock',item.get('quantity',0)) if item else 0
        if have:self.call('pantry.consume',item_id=item['id'],quantity=have)
        if required>have:self.shop(query,quantity=required-have)

    def send_last_draft(self):
        for call,result in reversed(self.history):
            if call['name']=='email.create_draft':return self.call('email.send',**call['arguments'])
        self.ask('email.send','to','subject','body')

    def shop_limit(self,query,quantity,max_price,shopping=False):
        p=self.lookup('ecommerce.search','products',query,{'query':query})
        if shopping:self.call('shopping_list.add',name=query,quantity=quantity)
        if p and p.get('price',float('inf'))<max_price:
            self.call('ecommerce.add_to_cart',product_id=p['id'],quantity=quantity)
            self.call('ecommerce.checkout')

    def stock_best_buy(self,quantity=100):
        holdings=self.read('stocks.get_portfolio').get('holdings',[])
        quotes=[self.call('stocks.get_quote',symbol=h['symbol']).get('quote',{}) for h in holdings]
        if quotes:
            q=max(quotes,key=lambda x:x.get('change_pct',-999))
            if q.get('symbol'):self.call('stocks.place_order',symbol=q['symbol'],side='buy',quantity=quantity)

    def stock_to_total(self,symbol,total):
        holdings=self.read('stocks.get_portfolio').get('holdings',[])
        h=next((x for x in holdings if x.get('symbol')==symbol),None)
        quantity=total-(h.get('quantity',0) if h else 0)
        if quantity>0:self.call('stocks.place_order',symbol=symbol,side='buy',quantity=quantity)

    def stock_note(self,query,title='Cổ phiếu'):
        hs=self.read('stocks.get_portfolio').get('holdings',[])
        h=self.choose(hs,query)
        if h:self.call('notes.create',title=title,content=json.dumps(h,ensure_ascii=False))

    def habit_all(self,date,exclude=None):
        hs=self.read('habit_tracker.list_habits').get('habits',[])
        for h in hs:
            if exclude and fold(exclude) in fold(h.get('name','')):continue
            logs=h.get('logs',[])
            if date in logs or h.get('last_logged')==date or h.get('last_date')==date:continue
            self.call('habit_tracker.log',habit_id=h['id'],date=date)

    def transit_topup(self,origin,destination,rides=20):
        routes=self.call('public_transit.find_routes',origin=origin,destination=destination).get('routes',[])
        if routes:
            r=routes[0]
            self.call('public_transit.topup_card',amount=rides*r.get('fare',7000))
            self.call('public_transit.next_departures',stop=r['board_at'],route=r['name'])

    def reply_last_email(self,body):
        emails=self.read('email.list').get('emails',[])
        if emails:self.call('email.reply',email_id=emails[-1]['id'],body=body)

    def cancel_ride(self):
        rides=self.read('ride_hailing.get_ride').get('rides',[])
        for r in rides:
            if r.get('status') not in ['cancelled','completed']:self.call('ride_hailing.cancel',ride_id=r['id'])

    def return_or_cancel(self,query):
        orders=self.read('ecommerce.order_status').get('orders',[])
        values=[o for o in orders if fold(query) in fold(json.dumps(o,ensure_ascii=False))]
        if len(values)==1:
            o=values[0]
            self.call('ecommerce.request_return',order_id=o['id'],reason='changed_mind') if o.get('status')=='delivered' else self.call('ecommerce.cancel_order',order_id=o['id'])

    def bank_transactions(self,query='tiết kiệm'):
        a=self.lookup('banking.get_balance','accounts',query)
        if a:self.call('banking.list_transactions',account_id=a['id'])

    def stock_sell_fraction(self,symbol,fraction=0.5,extra_price=0):
        hs=self.read('stocks.get_portfolio').get('holdings',[])
        h=next((h for h in hs if h.get('symbol')==symbol),None)
        if not h:return
        q=self.read('stocks.get_quote',symbol=symbol).get('quote',{})
        self.call('stocks.place_order',symbol=symbol,side='sell',quantity=max(1,int(h['quantity']*fraction)),price=q.get('price',h.get('price',0))+extra_price)

    def stock_losses_alert(self):
        hs=self.read('stocks.get_portfolio').get('holdings',[])
        for h in hs:
            if h.get('price',0)<h.get('avg_price',0):self.call('stocks.set_price_alert',symbol=h['symbol'],price=h['avg_price'],direction='above')

    def stock_loss_buy(self):
        accounts=self.read('banking.get_balance').get('accounts',[])
        a=self.choose(accounts,'thanh toán')
        quantity=200 if a and a.get('balance',0)>10000000 else 100
        hs=self.read('stocks.get_portfolio').get('holdings',[])
        for h in hs:
            if h.get('price',0)<h.get('avg_price',0):self.call('stocks.place_order',symbol=h['symbol'],side='buy',quantity=quantity)

    def stock_note_actions(self):
        import re
        notes=self.read('notes.list').get('notes',[])
        for note in notes:
            text=str(note.get('content',''))
            syms=re.findall(r'\bZ[A-Z]{1,4}\b',text)
            nums=re.findall(r'\b\d{2,3}[.,]\d{3}\b|\b\d{5,6}\b',text)
            if syms and nums:
                price=int(nums[-1].replace('.','').replace(',',''))
                for sym in syms:self.call('stocks.set_price_alert',symbol=sym,price=price,direction='below' if 'duoi' in fold(text) or 'giam' in fold(text) else 'above')

    def recipe_max(self,query):
        rs=self.call('recipes.search',query=query).get('recipes',[])
        if rs:self.call('recipes.save',recipe_id=max(rs,key=lambda r:r.get('servings',0))['id'])

    def grocery_quote_chat(self,query,chat_query):
        ps=self.call('grocery.search',query=query).get('products',[])
        p=self.choose(ps,query)
        if p:self.chat(chat_query,p['name']+' giá '+str(p.get('price'))+' đồng')

    def fuel_calendar(self,threshold=23500):
        prices=self.call('fuel_price.get_price',type='ron95').get('prices',[])
        high=prices and prices[0].get('price',0)>threshold
        self.call('calendar.create_event',title='Đổ xăng',start=self.when([0,20,0] if high else [1,6,0]))

    def bus_book(self,destination,date,seats=1,origin=None,depart=None):
        origin=origin or self.profile['city']
        prior=self.known('train_bus_tickets.search')
        trips=prior.get('trips',[]) if prior else []
        if not trips:trips=self.call('train_bus_tickets.search',origin=origin,destination=destination,date=date,type='bus').get('trips',[])
        trips=[t for t in trips if t.get('date')==date and t.get('type')=='bus' and t.get('seats_left',0)>=seats]
        if depart:trips=[t for t in trips if t.get('depart')==depart]
        if len(trips)==1:self.call('train_bus_tickets.book',trip_id=trips[0]['id'],seats=seats)
        elif trips:self.ask('train_bus_tickets.book','trip_id')

    def cancel_flight(self,date=None,destination=None,remind_before_day=False):
        bs=self.read('flights.my_bookings').get('bookings',[])
        for b in bs:
            if b.get('status')=='cancelled':continue
            f=b.get('flight',{})
            if date and f.get('date')!=date:continue
            if destination and f.get('destination')!=destination:continue
            self.call('flights.cancel',booking_id=b['id'])
            if remind_before_day and f.get('date'):
                d=datetime.fromisoformat(f['date']+'T17:00:00+07:00')-timedelta(days=1)
                self.reminder(d.isoformat(),'gọi khách')

    def pantry_switch(self,query,device_room='bếp'):
        item=self.lookup('pantry.list_items','items',query)
        if item and item.get('stock',0)>0:
            ds=self.call('smart_home.get_status',room=device_room).get('devices',[])
            p=next((x for x in ds if x.get('type')=='plug'),None)
            if p:self.call('smart_home.switch_set',device_id=p['id'],power='on')
        else:self.call('shopping_list.add',name=query)

    def music_add_current(self,playlist):
        tracks=self.read('music.list_tracks')
        p=self.choose(tracks.get('playlists',[]),playlist)
        previous=[c for c,r in self.history if c['name']=='music.play' and c['arguments'].get('track_id')]
        if p and previous:self.call('music.add_to_playlist',track_id=previous[-1]['arguments']['track_id'],playlist_id=p['id'])

    def transit_fare_topup(self,origin,destination,rides=20,max_fare=None):
        rs=self.call('public_transit.find_routes',origin=origin,destination=destination).get('routes',[])
        if rs:
            r=rs[0]
            if max_fare is None or r.get('fare',0)<max_fare:self.call('public_transit.topup_card',amount=rides*r.get('fare',7000))

    def notes_translate(self,query,lang='en'):
        n=self.lookup('notes.list','notes',query)
        if n:self.translate(n.get('content',''),lang=lang)

    def recipe_filter_save(self,query,max_time=40):
        result=self.known('recipes.search')
        if result is None:result=self.call('recipes.search',query=query)
        for r in result.get('recipes',[]):
            if r.get('time_min',999)<max_time:self.call('recipes.save',recipe_id=r['id'])

    def recipe_fastest(self,query):
        rs=self.call('recipes.search',query=query).get('recipes',[])
        if rs:self.call('recipes.get',recipe_id=min(rs,key=lambda r:r.get('time_min',999))['id'])

    def keep_alarm(self,query):
        alarms=self.read('alarm.list').get('alarms',[])
        tomorrow=(self.now+timedelta(days=1)).date().isoformat()
        for a in alarms:
            if a.get('time','').startswith(tomorrow) and fold(query) not in fold(a.get('label','')):
                self.call('alarm.update',alarm_id=a['id'],enabled=False)

    def flight_checkin(self,query):
        bs=self.read('flights.my_bookings').get('bookings',[])
        bs=[b for b in bs if fold(query) in fold(json.dumps(b,ensure_ascii=False)) and b.get('status')!='cancelled']
        if len(bs)==1:self.call('flights.check_in',booking_id=bs[0]['id'])
        else:self.ask('flights.check_in','booking_id')

    def bill_autopay(self,bill_type='electricity',threshold=1000000):
        bills=self.read('bill_payment.list_bills').get('bills',[])
        if any(b.get('type')==bill_type and b.get('amount',threshold)<threshold for b in bills):self.call('bill_payment.schedule_autopay',type=bill_type,enabled=True)

    def stock_alert_name(self,query,price=None,direction='above'):
        h=self.lookup('stocks.get_portfolio','holdings',query)
        if h:self.call('stocks.set_price_alert',symbol=h['symbol'],price=price or h['avg_price'],direction=direction)

    def save_place(self,query,label,near=None):
        p=self.lookup('maps.search_place','places',query,{'query':query,**({'near':near} if near else {})})
        if p:self.call('maps.save_place',place_id=p['id'],label=label)

    def checkout_event(self,title,offset=0,before=60):
        e=self.event(title,offset)
        if e:self.call('grocery.checkout',address=self.profile['home_address'],delivery_time=(datetime.fromisoformat(e['start'])-timedelta(minutes=before)).isoformat())
        else:self.ask('grocery.checkout','delivery_time')

    def flight_named(self,query,action='cancel'):
        bs=self.read('flights.my_bookings').get('bookings',[])
        bs=[b for b in bs if fold(query) in fold(json.dumps(b,ensure_ascii=False)) and b.get('status')!='cancelled']
        if len(bs)==1:self.call('flights.'+action,booking_id=bs[0]['id'])
        else:self.ask('flights.'+action,'booking_id')

    def cancel_pending_one(self,query=None):
        orders=self.read('ecommerce.order_status').get('orders',[])
        os=[o for o in orders if o.get('status') in ['placed','shipping'] and (query is None or fold(query) in fold(json.dumps(o,ensure_ascii=False)))]
        if os:self.call('ecommerce.cancel_order',order_id=os[0]['id'])

    def stock_equal_quantity(self,buy_symbol,reference,threshold,price):
        quote=self.call('stocks.get_quote',symbol=buy_symbol).get('quote',{})
        if quote.get('price',float('inf'))>=threshold:return
        hs=self.read('stocks.get_portfolio').get('holdings',[])
        h=next((x for x in hs if x.get('symbol')==reference),None)
        if h:self.call('stocks.place_order',symbol=buy_symbol,side='buy',quantity=h['quantity'],price=price)

    def stock_alert_winners(self):
        hs=self.read('stocks.get_portfolio').get('holdings',[])
        for h in hs:
            if h.get('price',0)>h.get('avg_price',0):self.call('stocks.set_price_alert',symbol=h['symbol'],price=h['avg_price'],direction='below')

    def stock_alert_largest_loss(self):
        hs=self.read('stocks.get_portfolio').get('holdings',[])
        if not hs:return
        h=min(hs,key=lambda h:(h.get('price',0)-h.get('avg_price',0))/max(1,h.get('avg_price',1)))
        if h.get('price',0)<h.get('avg_price',0):self.call('stocks.set_price_alert',symbol=h['symbol'],price=h['avg_price'],direction='above')

    def forecast_chat(self,query):
        fs=self.forecast()
        chats=self.read('chat_ott.list_chats').get('chats',[])
        c=self.choose(chats,query)
        if not c:return self.ask('chat_ott.call' if any(x.get('rain') for x in fs) else 'chat_ott.send_message','chat_id')
        if any(x.get('rain') for x in fs):self.call('chat_ott.call',chat_id=c['id'],video=True)
        else:self.call('chat_ott.send_message',chat_id=c['id'],content='Giữ nguyên lịch nhé')

    def cancel_alarm(self,query):
        alarms=self.read('alarm.list').get('alarms',[])
        values=[a for a in alarms if fold(query) in fold(a.get('label',''))]
        if len(values)==1:self.call('alarm.delete',alarm_id=values[0]['id'])
        else:self.ask('alarm.delete','alarm_id')

    def bank_fuel(self,query,threshold=21500,high=700000,low=600000):
        ps=self.call('fuel_price.get_price',type='ron95').get('prices',[])
        amount=high if ps and ps[0].get('price',0)>threshold else low
        self.bank([[query,amount]])

    def notes_append(self,query,content):
        n=self.lookup('notes.list','notes',query)
        if n:self.call('notes.append',note_id=n['id'],content=content)

    def pay_qr_restaurant(self,query,amount):
        r=self.lookup('food_delivery.list_restaurants','restaurants',query)
        if r and r.get('merchant_id'):self.call('ewallet.pay_qr',merchant_id=r['merchant_id'],amount=amount)
        else:self.ask('ewallet.pay_qr','merchant_id')

    def menu(self,query):
        r=self.lookup('food_delivery.list_restaurants','restaurants',query)
        if r:self.call('food_delivery.get_menu',restaurant_id=r['id'])

    def flight_checkin_earliest(self):
        bs=self.read('flights.my_bookings').get('bookings',[])
        bs=[b for b in bs if b.get('status')!='cancelled' and not b.get('checked_in')]
        if bs:
            b=min(bs,key=lambda b:(b.get('flight',{}).get('date',''),b.get('flight',{}).get('depart','')))
            self.call('flights.check_in',booking_id=b['id'])

    def alarms_before(self,hour=6,new_hour=6,new_minute=30):
        alarms=self.read('alarm.list').get('alarms',[])
        tomorrow=(self.now+timedelta(days=1)).date().isoformat()
        for a in alarms:
            if a.get('time','').startswith(tomorrow):
                t=datetime.fromisoformat(a['time'])
                if t.hour<hour:self.call('alarm.update',alarm_id=a['id'],time=t.replace(hour=new_hour,minute=new_minute).isoformat())

    def email_read_sender(self,query):
        es=self.read('email.list').get('emails',[])
        values=[e for e in es if fold(query) in fold(json.dumps(e,ensure_ascii=False))]
        if values:self.call('email.read',email_id=values[-1]['id'])

    def end_parking_task(self,query):
        sessions=self.read('parking.list_sessions').get('sessions',[])
        s=self.choose([s for s in sessions if s.get('status')!='ended'],query)
        if not s:return
        self.call('parking.end_session',session_id=s['id'])
        self.call('tasks.add',title='Thu tiền giữ xe của khách '+str(s.get('total',0))+' đồng')

    def stock_straddle(self,query,delta=5000):
        h=self.lookup('stocks.get_portfolio','holdings',query)
        if not h:return
        q=self.call('stocks.get_quote',symbol=h['symbol']).get('quote',{})
        price=q.get('price',h.get('price',0))
        self.call('stocks.set_price_alert',symbol=h['symbol'],price=price+delta,direction='above')
        self.call('stocks.set_price_alert',symbol=h['symbol'],price=price-delta,direction='below')

    def scene(self,query):
        ss=self.read('smart_home.get_status').get('scenes',[])
        s=self.choose(ss,query)
        if s:self.call('smart_home.scene_run',scene_id=s['id'])

    def weather_recipe(self,rain_query,dry_query):
        current=self.read('weather.get_current').get('current',[])
        rain=any('mua' in fold(c.get('condition','')) for c in current)
        self.recipe(rain_query if rain else dry_query,save=True)

    def trip_before(self,destination,date,arrive_before,seats=1):
        ts=self.call('train_bus_tickets.search',origin=self.profile['city'],destination=destination,date=date).get('trips',[])
        ts=[t for t in ts if t.get('arrive','')<=arrive_before and t.get('seats_left',0)>=seats]
        if ts:
            t=min(ts,key=lambda t:t.get('depart',''))
            self.call('train_bus_tickets.book',trip_id=t['id'],seats=seats)

    def gold_note(self,title,kind='ring'):
        prices=self.read('gold_price.get_price',type=kind).get('prices',[])
        if prices:self.notes_append(title,'Giá bán ra vàng '+kind+' tối nay: '+str(prices[0].get('sell'))+' '+str(prices[0].get('unit','')))

    def contact_info(self,query):
        cs=self.contacts(query)
        if len(cs)==1:self.call('contacts.get',contact_id=cs[0]['id'])
        else:self.ask('contacts.get','contact_id')

    def music_add(self,title,artist,playlist):
        result=self.read('music.list_tracks')
        tracks=[t for t in result.get('tracks',[]) if fold(t.get('title'))==fold(title) and fold(t.get('artist'))==fold(artist)]
        p=self.choose(result.get('playlists',[]),playlist)
        if len(tracks)==1 and p:self.call('music.add_to_playlist',track_id=tracks[0]['id'],playlist_id=p['id'])
        else:self.ask('music.add_to_playlist','track_id')

    def departures_notify(self,stop,query,after='09:00'):
        ds=self.call('public_transit.next_departures',stop=stop).get('departures',[])
        times=[]
        for d in ds:
            t=str(d.get('time',''))
            t=t[11:16] if 'T' in t else t[:5]
            if t>=self.now.strftime('%H:%M'):times.append(t)
        if times and min(times)>after:self.contact_action(query,'sms',content='Chị sẽ tới muộn nhé')

    def contacts(self,query):
        # Reuse only contacts that have actually appeared in conversation/tool results.
        def match(values):
            exact=[c for c in values if fold(c.get('name',''))==fold(query)]
            return exact or [c for c in values if fold(query) in fold(c.get('name','')+' '+c.get('relation',''))]
        for call,result in reversed(self.history):
            if call['name']=='contacts.search':
                found=match(result.get('contacts',[]))
                if found:
                    return found
        if not self.env.validate('contacts.search',{'query':query})['ok']:return []
        result=self.call('contacts.search',query=query)
        return match(result.get('contacts',[])) or result.get('contacts',[])

    def contact_action(self,query,action='phone',content=None,amount=None,relation=None):
        values=self.contacts(query)
        if relation:
            values=[c for c in values if fold(relation) in fold(c.get('relation',''))]
        tool={'phone':'phone.call','sms':'sms.send','wallet':'ewallet.transfer','topup':'ewallet.topup_mobile'}[action]
        if len(values)!=1:
            return self.ask(tool,'contact_id' if action!='topup' else 'phone')
        c=values[0]
        if action=='phone':return self.call(tool,contact_id=c['id'])
        if action=='sms':return self.call(tool,contact_id=c['id'],content=content)
        if amount is None:return self.ask(tool,'amount')
        if action=='topup':return self.call(tool,phone=c['phone'],amount=amount)
        return self.call(tool,contact_id=c['id'],amount=amount)

    def bank(self,transfers,fallback=True,reserve=0):
        accounts=self.read('banking.get_balance').get('accounts',[])
        beneficiaries=self.read('banking.list_beneficiaries').get('beneficiaries',[])
        account=next((x for x in accounts if 'thanh toan' in fold(x.get('name',''))),accounts[0] if accounts else None)
        total=sum(amount for name,amount in transfers)
        if account and account['balance']-total<reserve and fallback:
            account=next((x for x in accounts if 'tiet kiem' in fold(x.get('name',''))),account)
        if not account:return self.ask('banking.transfer','from_account_id')
        for name,amount in transfers:
            vals=[b for b in beneficiaries if fold(name) in fold(b.get('name',''))]
            if not vals: vals=[b for b in beneficiaries if all(w in fold(b.get('name','')) for w in fold(name).split())]
            if len(vals)!=1:self.ask('banking.transfer','beneficiary_id');continue
            self.call('banking.transfer',from_account_id=account['id'],beneficiary_id=vals[0]['id'],amount=amount)

    def bank_bills(self,transfers,reserve=0):
        bills=self.read('bill_payment.list_bills').get('bills',[])
        resolved=[]
        for spec in transfers:
            name,types,*rest=spec
            types=[types] if isinstance(types,str) else types
            amount=sum(b['amount'] for b in bills if b['type'] in types and not b.get('paid'))
            if types==['earliest']:
                candidates=sorted([b for b in bills if not b.get('paid')],key=lambda b:b.get('due_date',''))
                amount=candidates[0]['amount'] if candidates else 0
            if rest:amount=int(amount*rest[0])
            if amount:resolved.append([name,amount])
        return self.bank(resolved,reserve=reserve)

    def pay(self,bill_type,source='auto',notify=None,else_reminder=False):
        bills=self.read('bill_payment.list_bills').get('bills',[])
        b=next((b for b in bills if b.get('type')==bill_type and not b.get('paid')),None)
        if not b:return
        actual=source
        if source=='auto':
            balance=self.read('ewallet.get_balance').get('balance',0)
            actual='ewallet' if balance>=b['amount'] else 'bank'
        self.call('bill_payment.pay',bill_id=b['id'],source=actual)
        if actual=='bank' and else_reminder:
            self.reminder([0,20,0],'nạp tiền vào ví')
        elif notify:
            self.contact_action(notify,'sms',content='đã đóng tiền '+bill_type+' rồi')

    def forecast(self,offset=1):
        date=(self.now+timedelta(days=offset)).date().isoformat()
        return self.read('weather.get_forecast',date=date).get('forecast',[])

    def event(self,title,offset=1):
        date=(self.now+timedelta(days=offset)).date().isoformat()
        def select(events):
            values=[e for e in events if e.get('status')!='cancelled' and e.get('start','').startswith(date) and fold(title) in fold(e.get('title',''))]
            return min(values,key=lambda x:x['start']) if values else None
        for call,result in reversed(self.history):
            if call['name']=='calendar.list_events':
                e=select(result.get('events',[]))
                if e:return e
        return select(self.call('calendar.list_events').get('events',[]))

    def rain_event(self,title,action='cancel',shift=60,notify=None,channel='sms',offset=1,ride=None):
        forecasts=self.forecast(offset)
        rain=any(f.get('rain') for f in forecasts)
        if not rain and not ride:return
        e=self.event(title,offset)
        if not e:return self.ask('calendar.update_event' if action=='shift' else 'calendar.cancel_event','event_id')
        if not rain and ride:
            return self.call('ride_hailing.book',pickup=self.profile['home_address'],dropoff=e['location'],vehicle=ride.get('vehicle','car4'),schedule_time=(datetime.fromisoformat(e['start'])-timedelta(minutes=ride.get('early',30))).isoformat())
        if action=='cancel':self.call('calendar.cancel_event',event_id=e['id']);msg='Mai hủy lịch '+e['title']+' nhé'
        else:
            start=datetime.fromisoformat(e['start'])+timedelta(minutes=shift)
            end=datetime.fromisoformat(e['end'])+timedelta(minutes=shift)
            self.call('calendar.update_event',event_id=e['id'],start=start.isoformat(),end=end.isoformat())
            msg='mai '+e['title']+' dời sang '+start.strftime('%H:%M')+' nhé'
        if notify:self.contact_action(notify,channel,content=msg)

    def shift_event(self,title,minutes=30,offset=1,alarm_before=None,notify=None):
        e=self.event(title,offset)
        if not e:return self.ask('calendar.update_event','event_id')
        start=datetime.fromisoformat(e['start'])+timedelta(minutes=minutes)
        end=datetime.fromisoformat(e['end'])+timedelta(minutes=minutes)
        self.call('calendar.update_event',event_id=e['id'],start=start.isoformat(),end=end.isoformat())
        if alarm_before is not None:self.alarm((start-timedelta(minutes=alarm_before)).isoformat())
        if notify:self.contact_action(notify,'sms',content=e['title']+' dời sang '+start.strftime('%H:%M')+' nhé')

    def ride_event(self,title,offset=0,early=None,remind=None,vehicle='car4',pickup='work_address',content='mang laptop'):
        e=self.event(title,offset)
        if not e:return self.ask('ride_hailing.book','dropoff')
        kwargs=dict(pickup=self.profile[pickup],dropoff=e['location'],vehicle=vehicle)
        if early is not None:kwargs['schedule_time']=(datetime.fromisoformat(e['start'])-timedelta(minutes=early)).isoformat()
        self.call('ride_hailing.book',**kwargs)
        if remind is not None:
            time=datetime.fromisoformat(kwargs.get('schedule_time') or self.now.isoformat())-timedelta(minutes=remind)
            self.reminder(time.isoformat(),content)

    def smart(self,room=None,floor=None,power=None,brightness=None,exclude=None,climate=None,fan=None,only_power=None,only_brightness=None,except_room=None,brightness_below=None):
        devices=self.read('smart_home.get_status').get('devices',[])
        if power is not None or brightness is not None:
            lights=[d for d in devices if d.get('type')=='light' and (room is None or fold(d.get('room'))==fold(room)) and (floor is None or d.get('floor')==floor)]
            excluded={d.get('room') for d in devices if d.get('type')==exclude and (d.get('state',{}).get('power')=='on' or (exclude=='climate' and d.get('state',{}).get('mode','off')!='off'))}
            for d in lights:
                if d.get('room') in excluded:continue
                if except_room and fold(d.get('room'))==fold(except_room):continue
                if only_power is not None and d.get('state',{}).get('power')!=only_power:continue
                if only_brightness is not None and d.get('state',{}).get('brightness')!=only_brightness:continue
                if brightness_below is not None and d.get('state',{}).get('brightness',100)>=brightness_below:continue
                self.call('smart_home.light_set',device_id=d['id'],power=power,brightness=brightness)
        if fan:
            for d in devices:
                if d.get('type')=='fan' and (fan.get('room') is None or fold(d.get('room'))==fold(fan['room'])):
                    if fan.get('exclude_climate') and any(x.get('room')==d.get('room') and x.get('type')=='climate' and x.get('state',{}).get('mode','off')!='off' for x in devices):continue
                    self.call('smart_home.fan_set',device_id=d['id'],power=fan.get('power','on'),speed=fan.get('speed'))
        if climate:
            cs=[d for d in devices if d.get('type')=='climate' and fold(d.get('room'))==fold(climate['room'])]
            if cs:self.call('smart_home.climate_set',device_id=cs[0]['id'],mode=climate.get('mode','cool'),temperature=climate.get('temperature'))

    def flight_alarm(self,destination=None,before=120,after_arrival=None,night_before=None,on_day=None):
        values=self.read('flights.my_bookings').get('bookings',[])
        values=[b for b in values if b.get('status')!='cancelled']
        if destination:values=[b for b in values if b.get('flight',{}).get('destination')==destination]
        if len(values)!=1:return self.ask('alarm.create','time')
        f=values[0]['flight']
        base=datetime.fromisoformat(f['date']+'T'+f['depart']+('' if '+' in f['depart'] else '+07:00'))
        if night_before:
            base=(base-timedelta(days=1)).replace(hour=night_before[0],minute=night_before[1])
        elif on_day:
            base=base.replace(hour=on_day[0],minute=on_day[1])
        elif after_arrival is not None:
            base=datetime.fromisoformat(f['date']+'T'+f['arrive']+'+07:00')+timedelta(minutes=after_arrival)
        else:base-=timedelta(minutes=before)
        self.alarm(base.isoformat())

    def flight_book(self,destination,date,passengers=1,earliest=None,latest=None,arrive_before=None,choice='cheapest',seat_class=None,max_price=None,alarm_before=None,notify=None,code=None,parking=None,ride=None,calendar=False,airline=None):
        origin={'TP.HCM':'SGN','Hà Nội':'HAN','Đà Nẵng':'DAD'}[self.profile['city']]
        flights=self.call('flights.search',origin=origin,destination=destination,date=date).get('flights',[])
        if airline:flights=[f for f in flights if fold(airline) in fold(f.get('airline',''))]
        if earliest:flights=[f for f in flights if f['depart']>=earliest]
        if latest:flights=[f for f in flights if f['depart']<latest]
        if arrive_before:flights=[f for f in flights if f['arrive']<arrive_before]
        if max_price:flights=[f for f in flights if f['price']<max_price]
        if not flights:return
        if code:
            matches=[f for f in flights if fold(code).replace(' ','') in fold(json.dumps(f,ensure_ascii=False)).replace(' ','')]
            if not matches:return self.ask('flights.book','flight_id')
            f=matches[0]
        elif choice=='latest':f=max(flights,key=lambda f:f['depart'])
        elif choice=='earliest':f=min(flights,key=lambda f:f['depart'])
        else:f=min(flights,key=lambda f:(f['price'],f['depart']))
        self.call('flights.book',flight_id=f['id'],passengers=passengers,seat_class=seat_class)
        if parking:
            parking=dict(parking)
            if 'before' in parking:
                before=parking.pop('before')
                base=datetime.fromisoformat(date+'T'+f['depart']+'+07:00')-timedelta(minutes=before)
                parking['time']=base.isoformat()
            self.parking(**parking)
        if ride:
            self.call('ride_hailing.book',pickup=self.profile['home_address'],dropoff=ride.get('dropoff','Sân bay Nội Bài'),vehicle=ride.get('vehicle','car4'),schedule_time=self.when(ride['time']))
        if calendar:self.call('calendar.create_event',title='Lịch trình công tác',start=date+'T'+f['depart']+':00+07:00')
        if alarm_before is not None:
            base=datetime.fromisoformat(date+'T'+f['depart']+'+07:00')-timedelta(minutes=alarm_before)
            self.alarm(base.isoformat())
        if notify:
            chats=self.read('chat_ott.list_chats').get('chats',[])
            cs=[c for c in chats if fold(notify) in fold(c.get('name',''))]
            if len(cs)==1:self.call('chat_ott.send_message',chat_id=cs[0]['id'],content='Giờ bay '+f['depart']+' ngày '+date)
            else:self.ask('chat_ott.send_message','chat_id')

    def food_group(self,restaurant,item,group,include_self=False):
        chats=self.read('chat_ott.list_chats').get('chats',[])
        matches=[c for c in chats if fold(group) in fold(c.get('name',''))]
        if len(matches)!=1:return self.ask('food_delivery.place_order','items')
        members=matches[0].get('member_count')
        if members is None:
            values=matches[0].get('members',[])
            members=len(values) if isinstance(values,list) else int(values)
        quantity=members+1 if include_self else max(1,members)
        restaurants=self.read('food_delivery.list_restaurants').get('restaurants',[])
        rs=[r for r in restaurants if fold(restaurant) in fold(r.get('name',''))]
        if len(rs)!=1:return self.ask('food_delivery.place_order','restaurant_id')
        r=rs[0]
        menu=self.call('food_delivery.get_menu',restaurant_id=r['id']).get('menu',[])
        its=[x for x in menu if fold(item) in fold(x.get('name',''))]
        its=[x for x in its if fold(item)==fold(x.get('name',''))] or its
        if len(its)!=1:return self.ask('food_delivery.place_order','items')
        self.call('food_delivery.place_order',restaurant_id=r['id'],items=[{'item_id':its[0]['id'],'quantity':quantity}])

    def gold(self,kind='sjc',field='buy',threshold=0,operator='lt',time=None,content='đi mua vàng',transfer=None):
        prices=self.read('gold_price.get_price',type=kind).get('prices',[])
        if not prices:return
        value=prices[0].get(field,0)
        passed={'lt':value<threshold,'gt':value>threshold,'ge':value>=threshold,'le':value<=threshold}[operator]
        if passed and time:self.reminder(time,content)
        elif not passed and transfer:self.bank([transfer],fallback=False)

    def run(self,operation):
        op=dict(operation)
        kind=op.pop('op')
        if kind=='ask':return self.ask(op['tool'],*op['params'])
        if kind=='raw':
            kwargs=op.pop('arguments',{})
            kwargs={k:(self.profile['home_address'] if v=='$home' else self.profile['work_address'] if v=='$work' else v) for k,v in kwargs.items()}
            return self.call(op.pop('name'),**kwargs)
        return getattr(self,kind)(**op)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('plans',type=Path)
    p.add_argument('--output',type=Path,default=Path('reviewed_predictions.jsonl'))
    p.add_argument('--append',action='store_true')
    a=p.parse_args()
    items={x['id']:x for x in map(json.loads,(ROOT/'public_test/public_test.jsonl').read_text().splitlines())}
    plans=json.loads(a.plans.read_text())
    done=set()
    if a.output.exists():done={json.loads(line)['id'] for line in a.output.read_text().splitlines() if line.strip()}
    out_dir=ROOT/'outputs'/'work_sessions';out_dir.mkdir(parents=True,exist_ok=True)
    with a.output.open('a',encoding='utf-8') as stream:
        for sample_id,operations in plans.items():
            if sample_id in done:continue
            path=out_dir/(sample_id+'.pickle')
            if path.exists():
                with path.open('rb') as f:session=pickle.load(f)
                if session.get_tool_calls() and not hasattr(session,'_reviewed_operation_index') and not a.append:
                    raise ValueError(f'{sample_id} has existing calls; inspect before resuming')
            else:
                session=ToolEnv('public_test',sample_id,root=ROOT)
                session._reviewed_operation_index=0
            def save():
                with path.open('wb') as f:pickle.dump(session,f)
            save()
            solver=Solver(items[sample_id],session,save)
            start_index=0 if a.append else getattr(session,'_reviewed_operation_index',0)
            for index,operation in enumerate(operations[start_index:],start_index):
                solver.run(operation)
                session._reviewed_operation_index=index+1
                save()
            record={'id':sample_id,'tool_calls':session.get_tool_calls(),'state_hash':session.get_state_hash()}
            stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
            print(sample_id,json.dumps(record['tool_calls'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
