import re
from openai import OpenAI
import os

from geograph.constants import OPENAI_API_KEY

ONTOLOGY_SCHEMA_TURTLE = """
@base <http://example.org/geographbg#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix geographbg: <http://example.org/geographbg#> .

# Classes
geographbg:Местоположение a owl:Class ; rdfs:label "Местоположение"@bg .
geographbg:Език a owl:Class ; rdfs:label "Език"@bg .
geographbg:КоличественаСтойност a owl:Class ; rdfs:label "Количествена Стойност"@bg .
geographbg:МернаЕдиница a owl:Class ; rdfs:label "Мерна Единица"@bg .
geographbg:Организация a owl:Class ; rdfs:label "Организация"@bg .
geographbg:ПрироднаЗабележителност a owl:Class ; rdfs:label "Природна Забележителност"@bg ; rdfs:subClassOf geographbg:Местоположение .
geographbg:АдминистративнаЕдиница a owl:Class ; rdfs:label "Административна Единица"@bg ; rdfs:subClassOf geographbg:Местоположение .
geographbg:ЗащитенаТеритория a owl:Class ; rdfs:label "Защитена Територия"@bg ; rdfs:subClassOf geographbg:Местоположение .
geographbg:ИнфраструктуренОбект a owl:Class ; rdfs:label "Инфраструктурен Обект"@bg ; rdfs:subClassOf geographbg:Местоположение .
geographbg:ИсторическаОбласт a owl:Class ; rdfs:label "Историческа Област"@bg ; rdfs:subClassOf geographbg:Местоположение .
geographbg:ЕтнографскаОбласт a owl:Class ; rdfs:label "Етнографска Област"@bg ; rdfs:subClassOf geographbg:Местоположение .
geographbg:Местност a owl:Class ; rdfs:label "Местност"@bg ; rdfs:subClassOf geographbg:Местоположение .
geographbg:ГеографскиРегион a owl:Class ; rdfs:label "Географски Регион"@bg ; rdfs:subClassOf geographbg:Местоположение .
geographbg:КултуренОбект a owl:Class ; rdfs:label "Културен Обект"@bg ; rdfs:subClassOf geographbg:Местоположение .
geographbg:РелефнаФорма a owl:Class ; rdfs:label "Релефна Форма"@bg ; rdfs:subClassOf geographbg:ПрироднаЗабележителност .
geographbg:ВоденОбект a owl:Class ; rdfs:label "Воден Обект"@bg ; rdfs:subClassOf geographbg:ПрироднаЗабележителност .
geographbg:БиологиченОбект a owl:Class ; rdfs:label "Биологичен Обект"@bg ; rdfs:subClassOf geographbg:ПрироднаЗабележителност .
geographbg:Планина a owl:Class ; rdfs:label "Планина"@bg ; rdfs:subClassOf geographbg:РелефнаФорма .
geographbg:ПланинскиВръх a owl:Class ; rdfs:label "Планински Връх"@bg ; rdfs:subClassOf geographbg:РелефнаФорма .
geographbg:Пещера a owl:Class ; rdfs:label "Пещера"@bg ; rdfs:subClassOf geographbg:РелефнаФорма .
geographbg:Проход a owl:Class ; rdfs:label "Проход"@bg ; rdfs:subClassOf geographbg:РелефнаФорма .
geographbg:Равнина a owl:Class ; rdfs:label "Равнина"@bg ; rdfs:subClassOf geographbg:РелефнаФорма .
geographbg:Низина a owl:Class ; rdfs:label "Низина"@bg ; rdfs:subClassOf geographbg:РелефнаФорма .
geographbg:Котловина a owl:Class ; rdfs:label "Котловина"@bg ; rdfs:subClassOf geographbg:РелефнаФорма .
geographbg:Плато a owl:Class ; rdfs:label "Плато"@bg ; rdfs:subClassOf geographbg:РелефнаФорма .
geographbg:Хълм a owl:Class ; rdfs:label "Хълм"@bg ; rdfs:subClassOf geographbg:РелефнаФорма .
geographbg:Пролом a owl:Class ; rdfs:label "Пролом"@bg ; rdfs:subClassOf geographbg:РелефнаФорма .
geographbg:Долина a owl:Class ; rdfs:label "Долина"@bg ; rdfs:subClassOf geographbg:РелефнаФорма .
geographbg:Рид a owl:Class ; rdfs:label "Рид"@bg ; rdfs:subClassOf geographbg:РелефнаФорма .
geographbg:Циркус a owl:Class ; rdfs:label "Циркус"@bg ; rdfs:subClassOf geographbg:РелефнаФорма .
geographbg:Седловина a owl:Class ; rdfs:label "Седловина"@bg ; rdfs:subClassOf geographbg:РелефнаФорма .
geographbg:СкалноОбразувание a owl:Class ; rdfs:label "Скално Образувание"@bg ; rdfs:subClassOf geographbg:РелефнаФорма .
geographbg:Река a owl:Class ; rdfs:label "Река"@bg ; rdfs:subClassOf geographbg:ВоденОбект .
geographbg:Езеро a owl:Class ; rdfs:label "Езеро"@bg ; rdfs:subClassOf geographbg:ВоденОбект .
geographbg:Море a owl:Class ; rdfs:label "Море"@bg ; rdfs:subClassOf geographbg:ВоденОбект .
geographbg:Язовир a owl:Class ; rdfs:label "Язовир"@bg ; rdfs:subClassOf geographbg:ВоденОбект .
geographbg:Извор a owl:Class ; rdfs:label "Извор"@bg ; rdfs:subClassOf geographbg:ВоденОбект .
geographbg:Водопад a owl:Class ; rdfs:label "Водопад"@bg ; rdfs:subClassOf geographbg:ВоденОбект .
geographbg:Блато a owl:Class ; rdfs:label "Блато"@bg ; rdfs:subClassOf geographbg:ВоденОбект .
geographbg:Лагуна a owl:Class ; rdfs:label "Лагуна"@bg ; rdfs:subClassOf geographbg:ВоденОбект .
geographbg:МинераленИзвор a owl:Class ; rdfs:label "Минерален Извор"@bg ; rdfs:subClassOf geographbg:Извор .
geographbg:Поток a owl:Class ; rdfs:label "Поток"@bg ; rdfs:subClassOf geographbg:ВоденОбект .
geographbg:Държава a owl:Class ; rdfs:label "Държава"@bg ; rdfs:subClassOf geographbg:АдминистративнаЕдиница .
geographbg:Област a owl:Class ; rdfs:label "Област"@bg ; rdfs:subClassOf geographbg:АдминистративнаЕдиница .
geographbg:Община a owl:Class ; rdfs:label "Община"@bg ; rdfs:subClassOf geographbg:АдминистративнаЕдиница .
geographbg:НаселеноМясто a owl:Class ; rdfs:label "Населено Място"@bg ; rdfs:subClassOf geographbg:АдминистративнаЕдиница .
geographbg:Квартал a owl:Class ; rdfs:label "Квартал"@bg ; rdfs:subClassOf geographbg:АдминистративнаЕдиница .
geographbg:ВилнаЗона a owl:Class ; rdfs:label "Вилна Зона"@bg ; rdfs:subClassOf geographbg:АдминистративнаЕдиница .
geographbg:СтоличнаОбщина a owl:Class ; rdfs:label "Столична Община"@bg ; rdfs:subClassOf geographbg:Община .
geographbg:Град a owl:Class ; rdfs:label "Град"@bg ; rdfs:subClassOf geographbg:НаселеноМясто .
geographbg:Село a owl:Class ; rdfs:label "Село"@bg ; rdfs:subClassOf geographbg:НаселеноМясто .
geographbg:НационаленПарк a owl:Class ; rdfs:label "Национален Парк"@bg ; rdfs:subClassOf geographbg:ЗащитенаТеритория .
geographbg:ПрироденПарк a owl:Class ; rdfs:label "Природен Парк"@bg ; rdfs:subClassOf geographbg:ЗащитенаТеритория .
geographbg:Резерват a owl:Class ; rdfs:label "Резерват"@bg ; rdfs:subClassOf geographbg:ЗащитенаТеритория .
geographbg:ПоддържанРезерват a owl:Class ; rdfs:label "Поддържан Резерват"@bg ; rdfs:subClassOf geographbg:Резерват .
geographbg:ПрироднаЗабележителностСтатут a owl:Class ; rdfs:label "Природна Забележителност Статут"@bg ; rdfs:subClassOf geographbg:ЗащитенаТеритория .
geographbg:ЗащитенаМестност a owl:Class ; rdfs:label "Защитена Местност"@bg ; rdfs:subClassOf geographbg:ЗащитенаТеритория .
geographbg:БиосференРезерват a owl:Class ; rdfs:label "Биосферен Резерват"@bg ; rdfs:subClassOf geographbg:Резерват .
geographbg:Път a owl:Class ; rdfs:label "Път"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:ЖПЛиния a owl:Class ; rdfs:label "ЖП Линия"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Летище a owl:Class ; rdfs:label "Летище"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Пристанище a owl:Class ; rdfs:label "Пристанище"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Мост a owl:Class ; rdfs:label "Мост"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Тунел a owl:Class ; rdfs:label "Тунел"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:ВЕЦ a owl:Class ; rdfs:label "ВЕЦ"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Чешма a owl:Class ; rdfs:label "Чешма"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Беседка a owl:Class ; rdfs:label "Беседка"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Пътека a owl:Class ; rdfs:label "Пътека"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Улица a owl:Class ; rdfs:label "Улица"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Хотел a owl:Class ; rdfs:label "Хотел"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Ресторант a owl:Class ; rdfs:label "Ресторант"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Заслон a owl:Class ; rdfs:label "Заслон"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Хижа a owl:Class ; rdfs:label "Хижа"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:ТелевизионнаКула a owl:Class ; rdfs:label "Телевизионна Кула"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Каскада a owl:Class ; rdfs:label "Каскада"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Кариера a owl:Class ; rdfs:label "Кариера"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Полигон a owl:Class ; rdfs:label "Полигон"@bg .
geographbg:Баластриера a owl:Class ; rdfs:label "Баластриера"@bg .
geographbg:Водопровод a owl:Class ; rdfs:label "Водопровод"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Канал a owl:Class ; rdfs:label "Канал"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:СкиПиста a owl:Class ; rdfs:label "Ски Писта"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Лифт a owl:Class ; rdfs:label "Лифт"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Стадион a owl:Class ; rdfs:label "Стадион"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:База a owl:Class ; rdfs:label "База"@bg ; rdfs:subClassOf geographbg:ИнфраструктуренОбект .
geographbg:Крепост a owl:Class ; rdfs:label "Крепост"@bg ; rdfs:subClassOf geographbg:КултуренОбект .
geographbg:Манастир a owl:Class ; rdfs:label "Манастир"@bg ; rdfs:subClassOf geographbg:КултуренОбект .
geographbg:Паметник a owl:Class ; rdfs:label "Паметник"@bg ; rdfs:subClassOf geographbg:КултуренОбект .
geographbg:Музей a owl:Class ; rdfs:label "Музей"@bg ; rdfs:subClassOf geographbg:КултуренОбект .
geographbg:Парк a owl:Class ; rdfs:label "Парк"@bg ; rdfs:subClassOf geographbg:БиологиченОбект .
geographbg:Гора a owl:Class ; rdfs:label "Гора"@bg ; rdfs:subClassOf geographbg:БиологиченОбект .
geographbg:Градина a owl:Class ; rdfs:label "Градина"@bg ; rdfs:subClassOf geographbg:БиологиченОбект .
geographbg:Поле a owl:Class ; rdfs:label "Поле"@bg ; rdfs:subClassOf geographbg:БиологиченОбект .
geographbg:Ливада a owl:Class ; rdfs:label "Ливада"@bg ; rdfs:subClassOf geographbg:БиологиченОбект .
geographbg:ПолезноИзкопаемо a owl:Class ; rdfs:label "Полезно Изкопаемо"@bg .
geographbg:Почва a owl:Class ; rdfs:label "Почва"@bg .
geographbg:РастителенВид a owl:Class ; rdfs:label "Растителен Вид"@bg .
geographbg:ЖивотинскиВид a owl:Class ; rdfs:label "Животински Вид"@bg .
geographbg:КлиматичнаЗона a owl:Class ; rdfs:label "Климатична Зона"@bg .
geographbg:Събитие a owl:Class ; rdfs:label "Събитие"@bg .
geographbg:Период a owl:Class ; rdfs:label "Период"@bg .

# Object Properties
geographbg:държава a owl:ObjectProperty ; rdfs:label "държава"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range geographbg:Държава .
geographbg:находящСеВ a owl:ObjectProperty ; rdfs:label "находящСеВ"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range geographbg:Местоположение .
geographbg:столица a owl:ObjectProperty ; rdfs:label "столица"@bg ; rdfs:domain geographbg:Държава ; rdfs:range geographbg:Град .
geographbg:официаленЕзик a owl:ObjectProperty ; rdfs:label "официаленЕзик"@bg ; rdfs:domain geographbg:Държава ; rdfs:range geographbg:Език .
geographbg:вливаСеВ a owl:ObjectProperty ; rdfs:label "вливаСеВ"@bg ; rdfs:domain geographbg:ВоденОбект ; rdfs:range geographbg:ВоденОбект .
geographbg:извираОт a owl:ObjectProperty ; rdfs:label "извираОт"@bg ; rdfs:domain geographbg:ВоденОбект ; rdfs:range geographbg:Местоположение .
geographbg:притокНа a owl:ObjectProperty ; rdfs:label "притокНа"@bg ; rdfs:domain geographbg:Река ; rdfs:range geographbg:Река .
geographbg:граничиС a owl:ObjectProperty ; rdfs:label "граничиС"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range geographbg:Местоположение .
geographbg:областенЦентър a owl:ObjectProperty ; rdfs:label "областенЦентър"@bg ; rdfs:domain geographbg:Област ; rdfs:range geographbg:Град .
geographbg:общинскиЦентър a owl:ObjectProperty ; rdfs:label "общинскиЦентър"@bg ; rdfs:domain geographbg:Община ; rdfs:range geographbg:НаселеноМясто .
geographbg:намираСеНаРека a owl:ObjectProperty ; rdfs:label "намираСеНаРека"@bg ; rdfs:domain geographbg:ВоденОбект ; rdfs:range geographbg:Река .
geographbg:намираСеВЗемлищетоНа a owl:ObjectProperty ; rdfs:label "намираСеВЗемлищетоНа"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range geographbg:НаселеноМясто .
geographbg:разположенВМестност a owl:ObjectProperty ; rdfs:label "разположенВМестност"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range geographbg:Местност .
geographbg:частОтПрироденПарк a owl:ObjectProperty ; rdfs:label "частОтПрироденПарк"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range geographbg:ПрироденПарк .
geographbg:найВисокВръхНа a owl:ObjectProperty ; rdfs:label "найВисокВръхНа"@bg ; rdfs:domain geographbg:Планина ; rdfs:range geographbg:ПланинскиВръх .
geographbg:административенЦентърНа a owl:ObjectProperty ; rdfs:label "административенЦентърНа"@bg ; rdfs:domain geographbg:НаселеноМясто ; rdfs:range geographbg:АдминистративнаЕдиница .
geographbg:свързва a owl:ObjectProperty ; rdfs:label "свързва"@bg ; rdfs:domain geographbg:ИнфраструктуренОбект ; rdfs:range geographbg:Местоположение .
geographbg:преминаваПрез a owl:ObjectProperty ; rdfs:label "преминаваПрез"@bg ; rdfs:domain geographbg:ИнфраструктуренОбект, geographbg:Река ; rdfs:range geographbg:Местоположение .
geographbg:имаПриток a owl:ObjectProperty ; rdfs:label "имаПриток"@bg ; rdfs:domain geographbg:Река ; rdfs:range geographbg:Река .
geographbg:еСтолицаНа a owl:ObjectProperty ; rdfs:label "еСтолицаНа"@bg ; rdfs:domain geographbg:Град ; rdfs:range geographbg:Държава .
geographbg:еОбластенЦентърНа a owl:ObjectProperty ; rdfs:label "еОбластенЦентърНа"@bg ; rdfs:domain geographbg:Град ; rdfs:range geographbg:Област .
geographbg:еОбщинскиЦентърНа a owl:ObjectProperty ; rdfs:label "еОбщинскиЦентърНа"@bg ; rdfs:domain geographbg:НаселеноМясто ; rdfs:range geographbg:Община .
geographbg:имаИзлазНа a owl:ObjectProperty ; rdfs:label "имаИзлазНа"@bg ; rdfs:domain geographbg:АдминистративнаЕдиница, geographbg:Местоположение ; rdfs:range geographbg:ВоденОбект .
geographbg:частОтКаскада a owl:ObjectProperty ; rdfs:label "частОтКаскада"@bg ; rdfs:domain geographbg:ИнфраструктуренОбект ; rdfs:range geographbg:Каскада .
geographbg:стопанисваСеОт a owl:ObjectProperty ; rdfs:label "стопанисваСеОт"@bg ; rdfs:domain geographbg:Местоположение, geographbg:ИнфраструктуренОбект ; rdfs:range geographbg:Организация .
geographbg:надморскаВисочина a owl:ObjectProperty ; rdfs:label "надморскаВисочина"@bg ; rdfs:domain geographbg:Местоположение, geographbg:РелефнаФорма ; rdfs:range geographbg:КоличественаСтойност .
geographbg:дължинаВБългария a owl:ObjectProperty ; rdfs:label "дължинаВБългария"@bg ; rdfs:domain geographbg:Река ; rdfs:range geographbg:КоличественаСтойност .
geographbg:общаДължина a owl:ObjectProperty ; rdfs:label "общаДължина"@bg ; rdfs:domain geographbg:Река, geographbg:Път ; rdfs:range geographbg:КоличественаСтойност .
geographbg:площНаВодосборенБасейн a owl:ObjectProperty ; rdfs:label "площНаВодосборенБасейн"@bg ; rdfs:domain geographbg:Река ; rdfs:range geographbg:КоличественаСтойност .
geographbg:имаПлощ a owl:ObjectProperty ; rdfs:label "имаПлощ"@bg ; rdfs:domain geographbg:Местоположение, geographbg:АдминистративнаЕдиница, geographbg:ЗащитенаТеритория ; rdfs:range geographbg:КоличественаСтойност .
geographbg:имаНаселение a owl:ObjectProperty ; rdfs:label "имаНаселение"@bg ; rdfs:domain geographbg:АдминистративнаЕдиница, geographbg:НаселеноМясто ; rdfs:range geographbg:КоличественаСтойност .
geographbg:имаВисочинаНаПада a owl:ObjectProperty ; rdfs:label "имаВисочинаНаПада"@bg ; rdfs:domain geographbg:Водопад ; rdfs:range geographbg:КоличественаСтойност .
geographbg:имаОбем a owl:ObjectProperty ; rdfs:label "имаОбем"@bg ; rdfs:domain geographbg:ВоденОбект ; rdfs:range geographbg:КоличественаСтойност .
geographbg:имаЗалятаПлощ a owl:ObjectProperty ; rdfs:label "имаЗалятаПлощ"@bg ; rdfs:domain geographbg:ВоденОбект ; rdfs:range geographbg:КоличественаСтойност .
geographbg:имаСреднаНадморскаВисочина a owl:ObjectProperty ; rdfs:label "имаСреднаНадморскаВисочина"@bg ; rdfs:domain geographbg:РелефнаФорма ; rdfs:range geographbg:КоличественаСтойност .
geographbg:имаМаксималнаДълбочина a owl:ObjectProperty ; rdfs:label "имаМаксималнаДълбочина"@bg ; rdfs:domain geographbg:ВоденОбект ; rdfs:range geographbg:КоличественаСтойност .
geographbg:имаДължинаНаСтена a owl:ObjectProperty ; rdfs:label "имаДължинаНаСтена"@bg ; rdfs:domain geographbg:Язовир ; rdfs:range geographbg:КоличественаСтойност .
geographbg:имаВисочинаНаСтена a owl:ObjectProperty ; rdfs:label "имаВисочинаНаСтена"@bg ; rdfs:domain geographbg:Язовир ; rdfs:range geographbg:КоличественаСтойност .
geographbg:имаКапацитет a owl:ObjectProperty ; rdfs:label "имаКапацитет"@bg ; rdfs:domain geographbg:ИнфраструктуренОбект ; rdfs:range geographbg:КоличественаСтойност .
geographbg:имаДебит a owl:ObjectProperty ; rdfs:label "имаДебит"@bg ; rdfs:domain geographbg:ВоденОбект ; rdfs:range geographbg:КоличественаСтойност .
geographbg:имаТемпература a owl:ObjectProperty ; rdfs:label "имаТемпература"@bg ; rdfs:domain geographbg:Извор ; rdfs:range geographbg:КоличественаСтойност .
geographbg:имаМернаЕдиница a owl:ObjectProperty ; rdfs:label "имаМернаЕдиница"@bg ; rdfs:domain geographbg:КоличественаСтойност ; rdfs:range geographbg:МернаЕдиница .
geographbg:имаПолезноИзкопаемо a owl:ObjectProperty ; rdfs:label "имаПолезноИзкопаемо"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range geographbg:ПолезноИзкопаемо .
geographbg:имаТипПочва a owl:ObjectProperty ; rdfs:label "имаТипПочва"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range geographbg:Почва .
geographbg:имаХарактеренРастителенВид a owl:ObjectProperty ; rdfs:label "имаХарактеренРастителенВид"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range geographbg:РастителенВид .
geographbg:имаХарактеренЖивотинскиВид a owl:ObjectProperty ; rdfs:label "имаХарактеренЖивотинскиВид"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range geographbg:ЖивотинскиВид .
geographbg:принадлежиКъмКлиматичнаЗона a owl:ObjectProperty ; rdfs:label "принадлежиКъмКлиматичнаЗона"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range geographbg:КлиматичнаЗона .
geographbg:еСъставящаЧастНа a owl:ObjectProperty ; rdfs:label "еСъставящаЧастНа"@bg ; rdfs:domain geographbg:Река ; rdfs:range geographbg:Река .
geographbg:имаСъставящаЧаст a owl:ObjectProperty ; rdfs:label "имаСъставящаЧаст"@bg ; rdfs:domain geographbg:Река ; rdfs:range geographbg:Река .
geographbg:свързанС a owl:ObjectProperty ; rdfs:label "свързанС"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range geographbg:Местоположение .
geographbg:захранваСеОт a owl:ObjectProperty ; rdfs:label "захранваСеОт"@bg ; rdfs:domain geographbg:ВоденОбект ; rdfs:range geographbg:ВоденОбект .
geographbg:близоДо a owl:ObjectProperty ; rdfs:label "близоДо"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range geographbg:Местоположение .
geographbg:частОтИнфраструктуренОбект a owl:ObjectProperty ; rdfs:label "частОтИнфраструктуренОбект"@bg ; rdfs:domain geographbg:ИнфраструктуренОбект ; rdfs:range geographbg:ИнфраструктуренОбект .

# Datatype Properties
geographbg:имаСтойност a owl:DatatypeProperty ; rdfs:label "имаСтойност"@bg ; rdfs:domain geographbg:КоличественаСтойност ; rdfs:range xsd:decimal .
geographbg:имаГеографскаШирина a owl:DatatypeProperty ; rdfs:label "имаГеографскаШирина"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range xsd:decimal .
geographbg:имаГеографскаДължина a owl:DatatypeProperty ; rdfs:label "имаГеографскаДължина"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range xsd:decimal .
geographbg:имаКлимат a owl:DatatypeProperty ; rdfs:label "имаКлимат"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range xsd:string .
geographbg:имаГеология a owl:DatatypeProperty ; rdfs:label "имаГеология"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range xsd:string .
geographbg:имаФлора a owl:DatatypeProperty ; rdfs:label "имаФлора"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range xsd:string .
geographbg:имаФауна a owl:DatatypeProperty ; rdfs:label "имаФауна"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range xsd:string .
geographbg:имаАлтернативноИме a owl:DatatypeProperty ; rdfs:label "имаАлтернативноИме"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range xsd:string .
geographbg:имаОписание a owl:DatatypeProperty ; rdfs:label "имаОписание"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range xsd:string .
geographbg:имаИстория a owl:DatatypeProperty ; rdfs:label "имаИстория"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range xsd:string .
geographbg:имаЕтимология a owl:DatatypeProperty ; rdfs:label "имаЕтимология"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range xsd:string .
geographbg:имаСтатутНаЗащитенаМестност a owl:DatatypeProperty ; rdfs:label "имаСтатутНаЗащитенаМестност"@bg ; rdfs:domain geographbg:ЗащитенаТеритория ; rdfs:range xsd:string .
geographbg:обявенаСъсЗаповед a owl:DatatypeProperty ; rdfs:label "обявенаСъсЗаповед"@bg ; rdfs:domain geographbg:ЗащитенаТеритория ; rdfs:range xsd:string .
geographbg:основанПрезГодина a owl:DatatypeProperty ; rdfs:label "основанПрезГодина"@bg ; rdfs:domain geographbg:Местоположение, geographbg:Организация, geographbg:КултуренОбект ; rdfs:range xsd:gYear .
geographbg:преименуванНа a owl:DatatypeProperty ; rdfs:label "преименуванНа"@bg ; rdfs:domain geographbg:Местоположение ; rdfs:range xsd:string .
geographbg:имаТипСтена a owl:DatatypeProperty ; rdfs:label "имаТипСтена"@bg ; rdfs:domain geographbg:Язовир ; rdfs:range xsd:string .
geographbg:датаНаОбявяване a owl:DatatypeProperty ; rdfs:label "датаНаОбявяване"@bg ; rdfs:domain geographbg:ЗащитенаТеритория ; rdfs:range xsd:date .
geographbg:датаНаЗавършване a owl:DatatypeProperty ; rdfs:label "датаНаЗавършване"@bg ; rdfs:domain geographbg:ИнфраструктуренОбект ; rdfs:range xsd:date .

# Inverse Properties
geographbg:имаПриток owl:inverseOf geographbg:притокНа .
geographbg:притокНа owl:inverseOf geographbg:имаПриток .
geographbg:еСтолицаНа owl:inverseOf geographbg:столица .
geographbg:столица owl:inverseOf geographbg:еСтолицаНа .
geographbg:имаСъставящаЧаст owl:inverseOf geographbg:еСъставящаЧастНа .
geographbg:еСъставящаЧастНа owl:inverseOf geographbg:имаСъставящаЧаст .

# Unit Instances
geographbg:Метър a geographbg:МернаЕдиница ; rdfs:label "m"@bg ; rdfs:comment "Символ: m"@bg .
geographbg:Километър a geographbg:МернаЕдиница ; rdfs:label "km"@bg ; rdfs:comment "Символ: km"@bg .
geographbg:КвадратенКилометър a geographbg:МернаЕдиница ; rdfs:label "km²"@bg ; rdfs:comment "Символ: km²"@bg .
geographbg:Хектар a geographbg:МернаЕдиница ; rdfs:label "ha"@bg ; rdfs:comment "Символ: ha"@bg .
geographbg:Декар a geographbg:МернаЕдиница ; rdfs:label "дка"@bg ; rdfs:comment "Символ: дка"@bg .
geographbg:КубиченМетър a geographbg:МернаЕдиница ; rdfs:label "m³"@bg ; rdfs:comment "Символ: m³"@bg .
geographbg:Година a geographbg:МернаЕдиница ; rdfs:label "г."@bg ; rdfs:comment "Символ: г."@bg .
geographbg:Секунда a geographbg:МернаЕдиница ; rdfs:label "s"@bg ; rdfs:comment "Символ: s"@bg .
geographbg:БройЖители a geographbg:МернаЕдиница ; rdfs:label "брой жители"@bg ; rdfs:comment "Символ: брой жители"@bg .
geographbg:ДушиНаКвКм a geographbg:МернаЕдиница ; rdfs:label "души/km²"@bg ; rdfs:comment "Символ: души/km²"@bg .
geographbg:ГрадусЦелзий a geographbg:МернаЕдиница ; rdfs:label "°C"@bg ; rdfs:comment "Символ: °C"@bg .
geographbg:КубМетърВСекунда a geographbg:МернаЕдиница ; rdfs:label "m³/s"@bg ; rdfs:comment "Символ: m³/s"@bg .
geographbg:ЛитърВСекунда a geographbg:МернаЕдиница ; rdfs:label "l/s"@bg ; rdfs:comment "Символ: l/s"@bg .
geographbg:Мегават a geographbg:МернаЕдиница ; rdfs:label "MW"@bg ; rdfs:comment "Символ: MW"@bg .
geographbg:Киловат a geographbg:МернаЕдиница ; rdfs:label "kW"@bg ; rdfs:comment "Символ: kW"@bg .
geographbg:Ват a geographbg:МернаЕдиница ; rdfs:label "W"@bg ; rdfs:comment "Символ: W"@bg .
geographbg:КвадратенМетър a geographbg:МернаЕдиница ; rdfs:label "m²"@bg ; rdfs:comment "Символ: m²"@bg .

# CORE PREDEFINED INSTANCES (Example - you'd have your actual list)
geographbg:България a geographbg:Държава ; rdfs:label "България"@bg .
geographbg:Марица a geographbg:Река ; rdfs:label "Марица"@bg .
geographbg:Дунав a geographbg:Река ; rdfs:label "Дунав"@bg .
geographbg:Искър a geographbg:Река ; rdfs:label "Искър"@bg .
geographbg:ТунджаРека a geographbg:Река ; rdfs:label "Тунджа (река)"@bg .
geographbg:Рила a geographbg:Планина ; rdfs:label "Рила"@bg .
geographbg:Пирин a geographbg:Планина ; rdfs:label "Пирин"@bg .
geographbg:СтараПланина a geographbg:Планина ; rdfs:label "Стара планина"@bg .
geographbg:София a geographbg:Град ; rdfs:label "София"@bg .
geographbg:НякаквоБлато a geographbg:Блато, geographbg:ЗащитенаМестност ; rdfs:label "Някакво блато"@bg .
"""

ISKRETSKA_REKA_EXAMPLE = """
geographbg:ИскрецкаРека a geographbg:Река ;
    rdfs:label "Искрецка река"@bg ;
    geographbg:държава geographbg:България ;
    geographbg:имаАлтернативноИме "Реката"@bg ;
    geographbg:извираОт geographbg:МалаПланина ;
    geographbg:вливаСеВ geographbg:Искър ;
    geographbg:притокНа geographbg:Искър ;
    geographbg:общаДължина [
        a geographbg:КоличественаСтойност ;
        geographbg:имаСтойност "22.0"^^xsd:decimal ;
        geographbg:имаМернаЕдиница geographbg:Километър
    ] ;
    geographbg:площНаВодосборенБасейн [
        a geographbg:КоличественаСтойност ;
        geographbg:имаСтойност "57.0"^^xsd:decimal ;
        geographbg:имаМернаЕдиница geographbg:КвадратенКилометър
    ] ;
    geographbg:имаДебит [
        a geographbg:КоличественаСтойност ;
        geographbg:имаСтойност "4.00"^^xsd:decimal ;
        geographbg:имаМернаЕдиница geographbg:КубМетърВСекунда
    ] .
"""

LLM_PROMPT_TEMPLATE = f"""
You are an expert RDF/OWL ontology engineer specializing in Bulgarian geography. Your task is to analyze the provided Wikipedia page content and extract information to generate new RDF triples in Turtle format. These triples must conform to the provided ontology schema.

**Input You Will Receive:**

1.  **Existing Ontology Schema:** A snippet of the current ontology in Turtle format. This defines the available classes (e.g., `geographbg:Планина`, `geographbg:Река`, `geographbg:Град`) and properties (e.g., `geographbg:находящСеВ`, `geographbg:притокНа`, `geographbg:надморскаВисочина`, `geographbg:имаСтойност`, `geographbg:имаМернаЕдиница`).
2.  **Wikipedia Page Title:** The Bulgarian title of the Wikipedia page being processed.
3.  **Wikipedia Page Content:** The full text content of that Wikipedia page.

**Your Goal:**

Generate ONLY new RDF triples in Turtle format that represent the information extracted from the **Wikipedia Page Content**, using the classes and properties defined in the **Existing Ontology Schema**.

**Output Requirements:**

1.  **Format:** Output ONLY the new RDF triples in Turtle format. Start with the necessary `@base` and `@prefix` declarations as shown in the schema. Do not include any conversational text, explanations, or apologies.
2.  **Relevance:** The triples must be directly derivable from the provided Wikipedia page content. Do not infer information not present.
3.  **Schema Adherence:**
    *   Use ONLY the classes and properties defined in the provided ontology schema.
    *   Do NOT redefine existing classes or properties.
    *   Do NOT output triples for entities if their local names appear in the "Known Instance Local Names" list and they are *already fully and accurately defined* in the ontology, unless the new page provides *additional distinct facts* for that existing entity using existing properties. Refer to the "Known Instance Local Names" list to see which entity local names are already known. (You may need to infer their types from the schema or context if creating new triples for them).
4.  **Base URI and Prefixes:**
    *   All new resources you create (instances) MUST use the base URI: `http://example.org/geographbg#`
    *   Use the prefix `geographbg:` for this namespace.
    *   Use standard prefixes like `rdf:`, `rdfs:`, `owl:`, `xsd:` as needed and as shown in the schema.
5.  **Instance Naming:**
    *   For the primary entity described by the Wikipedia page, create an IRI using the `geographbg:` prefix and a CamelCase version of the **Wikipedia Page Title** (e.g., if the title is "Батак (язовир)", the instance might be `geographbg:БатакЯзовир`). Remove parentheses and special characters.
    *   For other new entities mentioned on the page (e.g., a river it flows into, a mountain it's on, a nearby village not yet in the ontology), create new IRIs similarly (e.g., `geographbg:ИмеНаОбект`).
    *   If an entity name might be ambiguous (e.g., "Стара река" can refer to multiple rivers), try to add a disambiguator based on context if possible (e.g., `geographbg:СтараРекаПритокСтряма`).
6.  **Labels:**
    *   For every new instance you create, add an `rdfs:label` with the Bulgarian name of the entity and the language tag `@bg`. Example: `geographbg:БатакЯзовир rdfs:label "Батак (язовир)"@bg .`
7.  **Quantitative Values (Critical):**
    *   For numerical data like altitude, length, area, volume, population, temperature, capacity, flow rate, etc., you MUST use the `geographbg:КоличественаСтойност` structure.
    *   Create a blank node or a new named instance of `geographbg:КоличественаСтойност`.
    *   Link the main entity to this `geographbg:КоличественаСтойност` instance using the appropriate object property (e.g., `geographbg:надморскаВисочина`, `geographbg:общаДължина`).
    *   On the `geographbg:КоличественаСтойност` instance:
        *   Use the datatype property `geographbg:имаСтойност` to specify the numerical value, typed with an appropriate XSD datatype (e.g., `"2925.0"^^xsd:decimal`, `"100"^^xsd:integer`).
        *   Use the object property `geographbg:имаМернаЕдиница` to link to the correct unit instance from the ontology (e.g., `geographbg:Метър`, `geographbg:Километър`, `geographbg:КвадратенКилометър`, `geographbg:Хектар`, `geographbg:КубиченМетър`, `geographbg:ГрадусЦелзий`, `geographbg:КубМетърВСекунда`).
8.  **Relationships:**
    *   Identify relationships between entities (e.g., a river flowing into another, a peak being part of a mountain, a village being in a municipality).
    *   Use the object properties from the schema to represent these (e.g., `geographbg:притокНа`, `geographbg:находящСеВ`, `geographbg:вливаСеВ`, `geographbg:частОтКаскада`).
    *   If an entity mentioned (e.g., "река Марица") has a local name that appears in the "Known Instance Local Names" list (e.g., `Марица`), use the existing IRI `geographbg:Марица`. Otherwise, create a new instance for it if it's described with new properties.
9.  **Datatype Properties:**
    *   Use datatype properties like `geographbg:имаАлтернативноИме`, `geographbg:имаОписание`, `geographbg:основанПрезГодина` (with `xsd:gYear`) where appropriate, based on the page content.
10. **Multiple Types:**
    *   If an entity clearly fits multiple classes (e.g., a `geographbg:Блато` that is also a `geographbg:ЗащитенаМестност`), declare it as an instance of all relevant classes:
        `geographbg:ИмеНаБлато a geographbg:Блато, geographbg:ЗащитенаМестност .`
11. **Focus on Facts:** Extract factual data that fits the ontology structure. Avoid highly interpretative statements or information that cannot be clearly mapped to existing properties.
12. **Bulgarian Language:** All `rdfs:label` values and string literals extracted from Bulgarian text should use the `@bg` language tag.

**Example of how to structure a quantitative value:**
```turtle
geographbg:Мусала geographbg:надморскаВисочина [
    a geographbg:КоличественаСтойност ;
    geographbg:имаСтойност "2925.0"^^xsd:decimal ;
    geographbg:имаМернаЕдиница geographbg:Метър
] .

Now, process the following Wikipedia page based on the provided ontology schema and known instances.

{{ONTOLOGY_AND_INSTANCES_SECTION}}

Wikipedia Page Title:
{{WIKIPEDIA_PAGE_TITLE}}

Wikipedia Page Content:
{{WIKIPEDIA_PAGE_CONTENT_TEXT}}

Expected Output (New Triples Only):
@base <http://example.org/geographbg#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix geographbg: <http://example.org/geographbg#> .

# FILL this section with the new triples
"""

def generate_existing_instances_list_turtle(ontology_file_path: str):
    """
    Parses a Turtle ontology file and extracts IRI + type for all instances
    belonging to the geographbg namespace.
    Outputs them as simple 'instance a Type .' statements.
    """
    local_names_set = set()
    try:
        with open(ontology_file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Ontology file not found at {ontology_file_path}")
        return []
    except Exception as e:
        print(f"Error reading ontology file: {e}")
        return []
    pattern = re.compile(r"^(geographbg:(\S+))\s+a\s+(geographbg:\S+(?:\s*,\s*geographbg:\S+)*)\s*[.;]", re.MULTILINE)
    for match in pattern.finditer(content):
        instance_local_name = match.group(2)
        local_names_set.add(instance_local_name)

    return sorted(list(local_names_set))

OPENAI_MODEL = "gpt-4"
OPENAI_ONTOLOGY_SYSTEM_CONTENT = ("You are an expert RDF/OWL ontology engineer. Your task is to extract information from"
                                  " Wikipedia content and generate RDF triples in Turtle format according to the provided schema.")


def llm_process_single_page(current_llm_prompt: str, wiki_page_title: str):
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    current_messages = [
        {"role": "system", "content": OPENAI_ONTOLOGY_SYSTEM_CONTENT},
        {"role": "user", "content": current_llm_prompt},
    ]

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=current_messages,
            max_tokens=2000,
            temperature=0.0,
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        error_message = f"Error processing page: {wiki_page_title}. Details: {e}"
        return error_message


if __name__ == "__main__":
    filepath = "ontologies/huge_ontology_v3.ttl"
    instances = generate_existing_instances_list_turtle(filepath)
    formatted_instance_local_names = "\n".join([f"- {name}" for name in instances])
    if not formatted_instance_local_names:
        formatted_instance_local_names = "(No specific instance local names provided for this run)"
    ontology_and_instances_section = f"""
    Full Ontology Schema:
    {ONTOLOGY_SCHEMA_TURTLE.strip()}
    
    Known Instance Local Names (these entities likely already exist in the ontology; use their existing IRIs like geographbg:LocalName):
    {formatted_instance_local_names}
    
    Illustrative Example of Detailed Extraction (for context, do not replicate unless input matches):
    {ISKRETSKA_REKA_EXAMPLE.strip()}
    """
    wiki_page_title = "ГКПП Ивайловград - Кипринос"
    final_llm_prompt_string = LLM_PROMPT_TEMPLATE.format(
        ONTOLOGY_AND_INSTANCES_SECTION=ontology_and_instances_section,
        WIKIPEDIA_PAGE_TITLE=wiki_page_title,
        WIKIPEDIA_PAGE_CONTENT_TEXT="Граничен контролно-пропускателен пункт Ивайловград – Кипринос се намира в близост до село Славеево, отвъд границата е село Саръхадър прекръстено от гърците на Кипринос. Пунктът свързва Ивайловград с Димотика. Открит на 9 септември 2010 година по случай 130-ата годишнина от установяването на дипломатически отношения между България и Гърция. Към тази дата това е петият пункт на българо-гръцката граница. На церемонията по откриването му присъстват президентите на България и Гърция Георги Първанов и Каролос Папуляс. Разкриването на граничния пункт съкращава разстоянието между Ивайловград и Свиленград с около 20 километра, тъй като пътят от Ивайловград до Свиленград през Гърция е по-кратък и сравнително равнинен, отколкото през България. От 1 януари 2025 г. ГКПП се закрива. Преминава се без спиране на автомобила и показване на документи на вече бившия ГКПП, но със съобразена скорост, означена с пътни знаци. Сградният фонд и инфраструктурата се запазват, така че при необходимост да могат да бъдат отворени. Граничните полицаи имат право да спират за проверки на случаен принцип и с оглед анализ на риска. == Източници ==",
    )
    response = llm_process_single_page(final_llm_prompt_string, wiki_page_title)
    print(instances)
