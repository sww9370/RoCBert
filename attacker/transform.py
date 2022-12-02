"""
将给定的token变型/替换成另一个字、拼音、符号等
"""
import numpy as np
import re
from collections import defaultdict
from pypinyin import pinyin, lazy_pinyin, Style
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
logger = logging.getLogger(__name__)



def is_alpha(s, alpha_pattern='^[a-zA-Z\\s012⚬θαενγ_|`~-]+$'):  # pattern中加入了自己新增的特殊替换符号以及sep符号
  return re.match(alpha_pattern, s) is not None


class Transform():
  def __init__(self, debug=False):
    self.debug = debug
    self.transformed_tokens = []
    self.mean_scores = []  # 每次处理text时的平均分，不清空
    self.max_scores = []  # 每次处理text时的最大分，不清空

  def clear(self):
    self.transformed_tokens = []

  def multi_ptr_trans(self, tokens, indices):
    new_tokens = tokens[:]
    for idx in indices:
      new_tokens[idx] = self._transform(new_tokens[idx])
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def _transform(self, target_token):
    raise NotImplementedError


class PhoneticTransform(Transform):
  def __init__(self, first_letter=False):
    """
    拼音变换
    """
    super().__init__()
    self.first_letter = first_letter

  def __call__(self, tokens, idx):
    target_token = tokens[idx]
    if idx > 0:
      left_token = tokens[idx - 1]
      if is_alpha(left_token):
        return None
    if idx + 1 < len(tokens):
      right_token = tokens[idx + 1]
      if is_alpha(right_token):
        return None

    new_token = self._transform(target_token)
    if new_token == target_token:
      return None

    new_tokens = tokens[:]
    new_tokens[idx] = new_token
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def _transform(self, target_token):
    try:
      py = lazy_pinyin(target_token)
      if self.first_letter:
        py = [s[0] for s in py]
        py = ''.join(py)
      else:
        py = ' '.join(py)
    except:
      logger.error(f'PhoneticTransform error:{target_token}')
      py = target_token
    return py

  def multi_ptr_trans(self, tokens, indices):
    new_tokens = tokens[:]
    for idx in indices:
      new_tokens[idx] = self._transform(new_tokens[idx])
    return new_tokens


class HuoXingWenTransform(Transform):
  def __init__(self):
    """
    火星文替换器
    """
    super().__init__()
    HXW = "啊阿埃挨哎唉哀皚癌藹矮艾礙愛隘鞍氨咹俺按暗岸胺案肮昂盎凹敖熬翱襖傲奧懊澳芭捌扒叭吧笆疤巴拔跋靶紦耙壩霸罷爸苩柏百擺佰敗拜稗斑癍搬扳般頒板蝂扮拌伴瓣半か絆邦幫梆榜膀綁棒磅蚌鎊傍謗苞胞包褒剝薄雹保堡飽寶菢報暴豹鮑爆杯碑悲卑丠輩褙贔鋇倍狽備憊焙被奔苯夲笨崩繃甭泵蹦迸逼鼻仳鄙筆彼碧蓖蔽畢斃毖幣庇痹閉敝弊必辟壁臂避陛鞭邊編貶扁便變卞辨辯辮遍標彪膘表鱉憋別癟彬斌瀕濱賓擯兵栤柄丙秉餅炳疒並箥菠播撥缽波博葧搏鉑箔伯帛舶脖膊渤泊駁捕卜哺補埠鈈咘步簿蔀怖擦猜裁材才財睬踩采彩菜蔡餐參蠶殘慚慘燦蒼艙倉滄藏操糙槽曹草廁策側冊測層蹭插叉茬茶查碴搽察岔差詫拆柴豺攙摻蟬饞讒纏鏟產闡顫昌猖場嘗瑺長償腸廠敞暢唱倡超抄鈔朝嘲潮巢吵炒車扯撤掣徹澈郴臣辰塵晨忱沉陳趁襯撐稱城橙成呈乘程懲澄誠承逞騁秤吃癡持匙池遲弛馳恥齒侈尺赤翅斥熾充沖蟲崇寵抽酬疇躊稠愁籌仇綢瞅醜臭初絀櫥廚躇鋤雛滁除楚礎儲矗搐觸處揣〣穿椽傳船喘串瘡窗幢床闖創吹炊捶錘垂春椿醇唇淳純蠢戳綽疵茨磁雌辭慈瓷詞此刺賜佽聰蔥囪匆從叢湊粗醋簇促躥篡竄摧崔催脆瘁粹淬翠村存団磋撮搓措挫諎搭達答瘩咑夶槑歹傣戴帶殆玳貸袋待逮怠耽擔丼單鄲撣膽旦氮但憚淡誕彈蜑當擋黨蕩檔刀搗蹈倒島禱導箌稻悼噵盜德嘚啲蹬燈登等瞪凳鄧堤低滴迪敵笛狄滌翟嫡抵底地蒂第渧弟遞締顛掂滇碘點典靛墊電佃甸店惦奠澱殿碉叼雕凋刁掉吊釣調跌爹碟蝶迭諜疊丁盯叮釘頂鼎錠萣訂丟東冬董懂動棟侗恫凍洞兜抖鬥陡豆逗痘都督蝳犢獨讀堵睹賭杜鍍肚喥渡妒端短鍛段斷緞堆兌隊對墩噸蹲敦頓囤鈍盾遁掇哆哆奪垛躲朵跺舵剁惰墮蛾峨鵝俄額訛娥惡厄扼遏鄂餓恩洏ㄦ聑爾餌洱②貳發罰筏伐乏閥法琺藩帆番翻樊礬釩繁凡煩反返范販犯飯泛坊芳方肪房防妨仿訪紡放菲非啡飝肥匪誹吠肺廢沸費芬酚吩氛汾紛墳焚汾粉奮份忿憤糞豐葑楓蜂峰鋒闏瘋烽逢馮縫諷奉鳳佛否夫敷膚孵扶拂輻幅氟符伏俘垺浮涪鍢袱弗甫撫輔俯釜斧脯腑府腐赴副覆賦複傅付阜父腹負富訃附婦縛咐噶嘎該改概鈣蓋溉幹咁杆柑竿肝趕感稈敢贛岡剛鋼缸肛綱崗港杠篙皋高膏羔糕搞鎬稿告哥歌擱戈鴿胳疙割革葛格蛤閣隔鉻個各給根哏耕哽庚羹埂耿梗工攻功恭龔供躬公宮弓鞏汞拱貢囲鉤勾溝苟狗垢構購夠辜菇咕箍估沽孤姑鼓古蠱骨穀股故顧固雇刮瓜剮寡掛褂乖拐怪棺關官冠觀管館罐慣灌貫咣廣逛瑰規圭矽歸龜閨軌鬼詭癸桂櫃跪圚劊輥滾棍鍋郭國果裹過囧骸駭海氦亥害駭酣憨邯韓含涵寒函喊罕翰撼捍旱憾悍焊汗漢夯杭航壕嚎豪毫郝恏耗號浩呵喝荷菏核禾囷何匼盒貉閡河涸赫褐鶴賀嘿嫼痕很狠恨哼亨橫衡恒轟哄烘虹鴻洪宏弘紅喉侯猴吼厚候後呼乎忽瑚壺葫胡蝴狐糊鍸弧虤唬護互滬戶婲嘩囮猾滑畫劃囮話槐徊懷淮壞歡環桓還緩換患喚瘓豢煥渙宦幻荒慌黃磺蝗簧瑝凰惶煌晃幌恍謊咴揮輝徽恢蛔囙毀悔慧卉惠晦賄穢茴燴彙諱誨繪葷昏婚魂渾混豁活夥吙獲戓惑霍貨禍擊圾基機畸稽積箕肌饑跡噭譏雞姬績緝吉極棘輯籍集及ゑ疾汲即嫉級擠幾脊己薊技冀季伎祭劑悸濟寄寂計記既忌際繼紀嘉枷夾佳镓加莢頰賈甲鉀假稼價架駕嫁殲監堅尖箋間煎兼肩艱奸緘繭檢柬堿鹼揀撿簡儉剪減薦檻鑒踐賤見鍵箭件健艦劍餞漸濺澗建僵薑將漿江疆蔣槳獎講匠醬降蕉椒礁焦膠交郊澆驕嬌嚼攪鉸矯僥腳狡角餃繳絞剿教酵轎較叫窖揭接皆秸街階截劫節莖聙晶鯨京驚精粳經囲警景頸靜境敬鏡徑痙靖竟競淨炯窘揪究糾玖韭久灸九酒廄救舊臼舅咎就疚鞠拘狙疽居駒菊局咀矩舉沮聚拒據巨具距踞鋸俱句懼炬劇捐鵑娟倦眷卷絹撅攫抉掘倔爵桔傑捷睫竭潔結解姐戒藉芥堺借介疥誡屆巾筋斤金紟津襟緊錦僅謹進靳晉禁近燼浸盡勁荊兢覺決訣絕均菌鈞軍君峻俊竣浚郡駿喀咖鉲咯開揩楷凱慨刊堪勘坎砍看康慷糠扛抗亢炕考拷烤靠坷苛柯棵磕顆科殼咳鈳渴克刻愙課肯啃墾懇坑吭涳恐孔控摳ロ扣寇枯哭窟苦酷庫褲誇垮挎跨胯塊筷儈快寬款匡筐狂框礦眶曠況虧盔巋窺葵奎魁傀饋愧潰坤昆捆困括擴廓闊垃拉喇蠟臘辣啦萊唻賴藍婪欄攔籃闌蘭瀾讕攬覽懶纜爛濫琅榔狼廊郎朗浪撈勞牢咾佬姥酪烙澇勒圞雷鐳蕾磊累儡壘擂肋類淚棱楞冷厘梨犁黎籬狸離漓悝李裏鯉禮莉荔吏栗麗厲勵礫曆利傈例俐痢竝粒瀝隸仂璃哩倆聯蓮連鐮廉憐漣簾斂臉鏈戀煉練糧涼梁粱良両輛量晾煷諒撩聊僚療燎寥遼潦叻撂鐐廖料列裂烮劣獵琳林磷霖臨鄰鱗淋凜賃吝拎玲菱零齡鈴伶羚淩靈陵嶺領另囹溜琉榴硫餾留劉瘤鋶柳六龖聾嚨籠窿隆壟攏隴嘍婁摟簍漏陋蘆盧顱廬爐擄鹵虜魯麓碌露蕗賂麤潞祿錄陸戮驢呂鋁侶旅履屢縷慮氯律率濾綠巒攣孿灤卵亂掠略掄輪倫侖淪綸論蘿螺羅邏鑼籮騾裸落洛駱絡媽麻瑪碼螞骉罵嘛嗎埋買麥賣邁脈瞞饅蠻滿蔓曼慢漫謾芒茫吂氓忙莽貓茅錨毛矛鉚卯茂冒帽貌貿仫玫枚梅酶黴煤莈眉媒鎂烸媄昧寐妹媚闁悶們萌蒙檬盟錳猛夢孟眯醚靡糜迷謎彌米秘覓泌蜜密冪棉眠綿冕免勉娩緬面苗描瞄藐秒渺廟妙蔑滅囻抿皿敏憫閩朙螟鳴銘名命謬摸摹蘑模膜磨摩魔抹末莫墨默沫漠寞陌謀牟某拇牡畝姆毋墓暮幕募慕朩目睦牧穆拿哪呐鈉那娜納氖乃奶耐奈喃侽難囊撓腦惱鬧淖呢餒內嫩能妮霓倪苨胒擬伱匿膩逆溺蔫拈姩碾攆撚念娘釀茑尿捏聶孽齧鑷鎳涅您檸獰凝寧擰濘犇扭鈕紐膿濃農弄奴努怒囡暖虐瘧挪懦糯諾哦歐鷗毆藕嘔偶漚啪趴爬帕怕琶拍排牌徘湃派攀潘盤磐盼畔判叛乓龐旁耪胖拋咆刨炮袍跑泡呸胚培裴賠陪配佩沛噴盆砰抨烹澎彭蓬棚硼篷膨萠鵬捧碰坯砒霹批披劈琵毗啤脾疲皮匹痞僻屁譬篇偏爿騙飄漂瓢票撇瞥拼頻貧品聘乒坪蘋萍平憑瓶評屏坡潑頗嘙破魄迫粕剖撲鋪仆莆葡菩蒲埔樸圃普浦譜曝瀑期欺棲戚妻七淒漆柒沏其棋奇歧畦崎臍齊旗祈祁騎起豈乞企啟契砌器気迄棄汽泣訖掐洽牽扡釺鉛芉遷簽仟謙乾黔錢鉗前潛遣淺譴塹嵌欠歉槍嗆腔羌牆薔強搶橇鍬敲悄橋瞧喬僑巧鞘撬翹峭俏竅切茄且怯竊欽侵儭秦琴勤芹擒禽寢沁圊輕氫傾卿清擎晴氰情頃請慶瓊窮秋丘邱浗求囚酋泅趨區蛆曲軀屈驅渠取娶齲趣去圈顴權醛灥铨痊拳猋券勸缺炔瘸卻鵲榷確雀裙群然燃冉染瓤壤攘嚷讓饒擾繞惹熱壬仁囚忍韌任認刃妊紉扔仍ㄖ戎茸蓉榮融熔溶容絨冗揉柔禸茹蠕儒孺洳辱乳汝入褥軟阮蕊瑞銳閏潤若弱撒灑薩腮鰓塞賽三三傘散桑嗓喪搔騷掃嫂瑟銫澀森僧莎砂殺刹沙紗儍啥煞篩曬珊苫杉屾刪煽衫閃陝擅贍膳善汕扇繕墒傷商賞晌仩尚裳梢捎稍燒芍勺韶尐哨邵紹奢賒蛇舌舍赦攝射懾涉社設砷申呻伸身深娠紳神沈審嬸甚腎慎滲聲苼甥牲升繩渻盛剩勝聖師夨獅施濕詩屍虱┿石拾塒什喰蝕實識史矢使屎駛始式示壵卋柿倳拭誓逝勢昰嗜噬適仕侍釋飾氏市恃室視試收掱首垨壽授售受瘦獸蔬樞梳殊抒輸菽舒淑疏圕贖孰熟薯暑曙署蜀黍鼠屬術述樹束戍豎墅庶數漱恕刷耍摔衰甩帥栓拴霜雙爽誰沝睡稅吮瞬順舜詤碩朔爍斯撕嘶思私司絲迉肆寺嗣四伺似飼巳松聳慫頌送宋訟誦搜艘擻嗽蘇酥俗素速粟僳塑溯宿訴肅酸蒜算雖隋隨綏髓誶歲穗遂隧祟孫損筍蓑梭唆縮瑣索鎖所塌彵咜她塔獺撻蹋踏胎苔抬囼泰酞呔態汰坍攤貪癱灘壇檀痰潭譚談坦毯袒碳探歎炭湯塘搪堂棠膛唐糖倘躺淌趟燙掏濤滔絛萄桃逃淘陶討套特藤騰疼謄梯剔踢銻提題蹄啼體替嚏惕涕剃屜兲添填畾憇恬舔腆挑條迢眺跳貼鐵帖廳聽烴汀廷停亭庭挺艇通桐酮瞳哃銅彤童桶捅筒統痛偷投頭透凸禿突圖徒途塗屠汢吐兔湍團推穨腿蛻褪退吞屯臀拖托脫鴕陀馱駝橢妥拓唾挖哇蛙窪娃瓦襪歪外豌彎灣玩頑丸烷完碗挽晚皖惋宛婉萬腕汪迋亡枉網往旺望莣妄威巍微危韋違桅圍唯惟為濰維葦萎委偉偽尾緯未蔚菋畏胃喂魏位渭謂尉慰衛瘟溫蚊攵聞紋吻穩紊問嗡翁甕撾蝸渦窩莪斡臥握沃莁嗚鎢烏汙誣屋無蕪梧吾吳毋武五捂午舞伍侮塢戊霧晤粅勿務悟誤昔熙析覀硒矽晰嘻吸錫犧稀息希悉膝夕惜熄烯溪汐犀檄襲席習媳囍銑洗系隙戲細瞎蝦匣霞轄暇峽俠狹丅廈夏嚇掀鍁先仙鮮纖鹹賢銜舷閑涎弦嫌顯險哯獻縣腺餡羨憲陷限線相廂鑲馫箱襄湘鄉翔祥詳想響享項巷橡像姠潒蕭硝霄削哮囂銷消宵淆曉曉孝校肖嘯笑效楔些歇蠍鞋協挾攜邪斜脅諧寫械卸蟹懈泄瀉謝屑薪芯鋅欣辛噺忻惢信釁煋腥猩惺興刑型形邢荇醒圉杏性姓兄凶胸匈洶雄熊休修羞朽嗅鏽秀袖繡墟戌需虛噓須徐許蓄酗敘旭序畜恤絮婿緒續軒喧宣懸旋玄選癬眩絢靴薛學穴雪血勳熏循旬詢尋馴巡殉汛訓訊遜迅壓押鴉鴨吖丫芽牙蚜崖衙涯雅啞亜訝焉咽閹煙淹鹽嚴研蜒岩延訁顏閻燚沿奄掩眼衍演豔堰燕厭硯雁唁彥焰宴諺驗殃央鴦秧楊揚佯瘍羴洋陽氧仰癢養樣漾邀腰妖瑤搖堯遙窯謠姚咬舀藥偠耀椰噎耶爺野冶吔頁掖業旪曳腋夜液┅壹醫揖銥依伊衤頤夷遺移儀胰疑沂宜姨彝椅蟻倚巳乙矣鉯藝抑噫邑屹億役臆逸肄疫亦裔意毅憶図益溢詣議誼譯異翼翌繹茵蔭因殷喑陰姻吟銀淫寅飲尹引隱茚英櫻嬰鷹應纓瑩螢營熒蠅迎贏盈影穎硬映喲擁傭臃癰庸雍踴蛹詠泳湧詠恿勇鼡幽優悠憂尤由郵鈾猶油遊酉洧伖右佑釉誘又呦迂淤於盂榆虞愚輿餘俞逾鱻愉渝漁隅予娛雨與嶼禹宇語羽玊域芋鬱籲遇喻峪禦愈欲獄育譽浴寓裕預豫馭鴛淵冤え垣袁原援轅園員圓猿源緣遠苑願怨院曰約越躍鑰嶽粵仴悅閱耘雲鄖勻隕尣運蘊醞暈韻孕匝砸雜栽哉災宰載洅茬咱攢暫贊贓贓葬遭糟鑿藻棗早澡蚤躁噪造皂灶燥責擇則澤賊怎增憎曾贈紮喳渣劄軋鍘閘眨柵榨咋乍炸詐摘齋宅窄債寨瞻氈詹粘沾盞斬輾嶄展蘸棧占戰站湛綻樟嶂彰漳涨掌漲杖丈帳賬仗脹瘴障招昭找沼趙照罩兆肇召遮折哲蟄轍者鍺蔗這浙珍斟眞甄砧臻貞針偵枕疹診震振鎮陣蒸掙睜征猙爭怔整拯㊣政幀症鄭證芝枝支吱蜘知肢脂汁の織職直植殖執徝侄址指止趾呮旨紙志摯擲至致置幟峙制智秩稚質炙痔滯治窒ф盅忠鍾衷終種腫重仲眾舟周州洲謅粥軸肘帚咒皺宙晝驟珠株蛛朱豬諸誅逐竹燭煮拄矚囑主著柱助蛀貯鑄築住紸祝駐抓爪拽專磚轉撰賺篆樁莊裝妝撞壯狀椎錐縋贅墜綴諄准捉拙卓桌琢茁酌啄著灼濁茲咨資姿滋淄孜紫仔籽滓孓自漬芓鬃棕蹤宗綜總縱鄒赱奏揍租足卒族祖詛阻組鑽纂嘴醉朂罪尊遵昨咗佐柞做作唑座"
    JIAN = "啊阿埃挨哎唉哀皑癌蔼矮艾碍爱隘鞍氨安俺按暗岸胺案肮昂盎凹敖熬翱袄傲奥懊澳芭捌扒叭吧笆疤巴拔跋靶把耙坝霸罢爸白柏百摆佰败拜稗斑班搬扳般颁板版扮拌伴瓣半办绊邦帮梆榜膀绑棒磅蚌镑傍谤苞胞包褒剥薄雹保堡饱宝抱报暴豹鲍爆杯碑悲卑北辈背贝钡倍狈备惫焙被奔苯本笨崩绷甭泵蹦迸逼鼻比鄙笔彼碧蓖蔽毕毙毖币庇痹闭敝弊必辟壁臂避陛鞭边编贬扁便变卞辨辩辫遍标彪膘表鳖憋别瘪彬斌濒滨宾摈兵冰柄丙秉饼炳病并玻菠播拨钵波博勃搏铂箔伯帛舶脖膊渤泊驳捕卜哺补埠不布步簿部怖擦猜裁材才财睬踩采彩菜蔡餐参蚕残惭惨灿苍舱仓沧藏操糙槽曹草厕策侧册测层蹭插叉茬茶查碴搽察岔差诧拆柴豺搀掺蝉馋谗缠铲产阐颤昌猖场尝常长偿肠厂敞畅唱倡超抄钞朝嘲潮巢吵炒车扯撤掣彻澈郴臣辰尘晨忱沉陈趁衬撑称城橙成呈乘程惩澄诚承逞骋秤吃痴持匙池迟弛驰耻齿侈尺赤翅斥炽充冲虫崇宠抽酬畴踌稠愁筹仇绸瞅丑臭初出橱厨躇锄雏滁除楚础储矗搐触处揣川穿椽传船喘串疮窗幢床闯创吹炊捶锤垂春椿醇唇淳纯蠢戳绰疵茨磁雌辞慈瓷词此刺赐次聪葱囱匆从丛凑粗醋簇促蹿篡窜摧崔催脆瘁粹淬翠村存寸磋撮搓措挫错搭达答瘩打大呆歹傣戴带殆代贷袋待逮怠耽担丹单郸掸胆旦氮但惮淡诞弹蛋当挡党荡档刀捣蹈倒岛祷导到稻悼道盗德得的蹬灯登等瞪凳邓堤低滴迪敌笛狄涤翟嫡抵底地蒂第帝弟递缔颠掂滇碘点典靛垫电佃甸店惦奠淀殿碉叼雕凋刁掉吊钓调跌爹碟蝶迭谍叠丁盯叮钉顶鼎锭定订丢东冬董懂动栋侗恫冻洞兜抖斗陡豆逗痘都督毒犊独读堵睹赌杜镀肚度渡妒端短锻段断缎堆兑队对墩吨蹲敦顿囤钝盾遁掇哆多夺垛躲朵跺舵剁惰堕蛾峨鹅俄额讹娥恶厄扼遏鄂饿恩而儿耳尔饵洱二贰发罚筏伐乏阀法珐藩帆番翻樊矾钒繁凡烦反返范贩犯饭泛坊芳方肪房防妨仿访纺放菲非啡飞肥匪诽吠肺废沸费芬酚吩氛分纷坟焚汾粉奋份忿愤粪丰封枫蜂峰锋风疯烽逢冯缝讽奉凤佛否夫敷肤孵扶拂辐幅氟符伏俘服浮涪福袱弗甫抚辅俯釜斧脯腑府腐赴副覆赋复傅付阜父腹负富讣附妇缚咐噶嘎该改概钙盖溉干甘杆柑竿肝赶感秆敢赣冈刚钢缸肛纲岗港杠篙皋高膏羔糕搞镐稿告哥歌搁戈鸽胳疙割革葛格蛤阁隔铬个各给根跟耕更庚羹埂耿梗工攻功恭龚供躬公宫弓巩汞拱贡共钩勾沟苟狗垢构购够辜菇咕箍估沽孤姑鼓古蛊骨谷股故顾固雇刮瓜剐寡挂褂乖拐怪棺关官冠观管馆罐惯灌贯光广逛瑰规圭硅归龟闺轨鬼诡癸桂柜跪贵刽辊滚棍锅郭国果裹过哈骸孩海氦亥害骇酣憨邯韩含涵寒函喊罕翰撼捍旱憾悍焊汗汉夯杭航壕嚎豪毫郝好耗号浩呵喝荷菏核禾和何合盒貉阂河涸赫褐鹤贺嘿黑痕很狠恨哼亨横衡恒轰哄烘虹鸿洪宏弘红喉侯猴吼厚候后呼乎忽瑚壶葫胡蝴狐糊湖弧虎唬护互沪户花哗华猾滑画划化话槐徊怀淮坏欢环桓还缓换患唤痪豢焕涣宦幻荒慌黄磺蝗簧皇凰惶煌晃幌恍谎灰挥辉徽恢蛔回毁悔慧卉惠晦贿秽会烩汇讳诲绘荤昏婚魂浑混豁活伙火获或惑霍货祸击圾基机畸稽积箕肌饥迹激讥鸡姬绩缉吉极棘辑籍集及急疾汲即嫉级挤几脊己蓟技冀季伎祭剂悸济寄寂计记既忌际继纪嘉枷夹佳家加荚颊贾甲钾假稼价架驾嫁歼监坚尖笺间煎兼肩艰奸缄茧检柬碱硷拣捡简俭剪减荐槛鉴践贱见键箭件健舰剑饯渐溅涧建僵姜将浆江疆蒋桨奖讲匠酱降蕉椒礁焦胶交郊浇骄娇嚼搅铰矫侥脚狡角饺缴绞剿教酵轿较叫窖揭接皆秸街阶截劫节茎睛晶鲸京惊精粳经井警景颈静境敬镜径痉靖竟竞净炯窘揪究纠玖韭久灸九酒厩救旧臼舅咎就疚鞠拘狙疽居驹菊局咀矩举沮聚拒据巨具距踞锯俱句惧炬剧捐鹃娟倦眷卷绢撅攫抉掘倔爵桔杰捷睫竭洁结解姐戒藉芥界借介疥诫届巾筋斤金今津襟紧锦仅谨进靳晋禁近烬浸尽劲荆兢觉决诀绝均菌钧军君峻俊竣浚郡骏喀咖卡咯开揩楷凯慨刊堪勘坎砍看康慷糠扛抗亢炕考拷烤靠坷苛柯棵磕颗科壳咳可渴克刻客课肯啃垦恳坑吭空恐孔控抠口扣寇枯哭窟苦酷库裤夸垮挎跨胯块筷侩快宽款匡筐狂框矿眶旷况亏盔岿窥葵奎魁傀馈愧溃坤昆捆困括扩廓阔垃拉喇蜡腊辣啦莱来赖蓝婪栏拦篮阑兰澜谰揽览懒缆烂滥琅榔狼廊郎朗浪捞劳牢老佬姥酪烙涝勒乐雷镭蕾磊累儡垒擂肋类泪棱楞冷厘梨犁黎篱狸离漓理李里鲤礼莉荔吏栗丽厉励砾历利傈例俐痢立粒沥隶力璃哩俩联莲连镰廉怜涟帘敛脸链恋炼练粮凉梁粱良两辆量晾亮谅撩聊僚疗燎寥辽潦了撂镣廖料列裂烈劣猎琳林磷霖临邻鳞淋凛赁吝拎玲菱零龄铃伶羚凌灵陵岭领另令溜琉榴硫馏留刘瘤流柳六龙聋咙笼窿隆垄拢陇楼娄搂篓漏陋芦卢颅庐炉掳卤虏鲁麓碌露路赂鹿潞禄录陆戮驴吕铝侣旅履屡缕虑氯律率滤绿峦挛孪滦卵乱掠略抡轮伦仑沦纶论萝螺罗逻锣箩骡裸落洛骆络妈麻玛码蚂马骂嘛吗埋买麦卖迈脉瞒馒蛮满蔓曼慢漫谩芒茫盲氓忙莽猫茅锚毛矛铆卯茂冒帽貌贸么玫枚梅酶霉煤没眉媒镁每美昧寐妹媚门闷们萌蒙檬盟锰猛梦孟眯醚靡糜迷谜弥米秘觅泌蜜密幂棉眠绵冕免勉娩缅面苗描瞄藐秒渺庙妙蔑灭民抿皿敏悯闽明螟鸣铭名命谬摸摹蘑模膜磨摩魔抹末莫墨默沫漠寞陌谋牟某拇牡亩姆母墓暮幕募慕木目睦牧穆拿哪呐钠那娜纳氖乃奶耐奈南男难囊挠脑恼闹淖呢馁内嫩能妮霓倪泥尼拟你匿腻逆溺蔫拈年碾撵捻念娘酿鸟尿捏聂孽啮镊镍涅您柠狞凝宁拧泞牛扭钮纽脓浓农弄奴努怒女暖虐疟挪懦糯诺哦欧鸥殴藕呕偶沤啪趴爬帕怕琶拍排牌徘湃派攀潘盘磐盼畔判叛乓庞旁耪胖抛咆刨炮袍跑泡呸胚培裴赔陪配佩沛喷盆砰抨烹澎彭蓬棚硼篷膨朋鹏捧碰坯砒霹批披劈琵毗啤脾疲皮匹痞僻屁譬篇偏片骗飘漂瓢票撇瞥拼频贫品聘乒坪苹萍平凭瓶评屏坡泼颇婆破魄迫粕剖扑铺仆莆葡菩蒲埔朴圃普浦谱曝瀑期欺栖戚妻七凄漆柒沏其棋奇歧畦崎脐齐旗祈祁骑起岂乞企启契砌器气迄弃汽泣讫掐洽牵扦钎铅千迁签仟谦乾黔钱钳前潜遣浅谴堑嵌欠歉枪呛腔羌墙蔷强抢橇锹敲悄桥瞧乔侨巧鞘撬翘峭俏窍切茄且怯窃钦侵亲秦琴勤芹擒禽寝沁青轻氢倾卿清擎晴氰情顷请庆琼穷秋丘邱球求囚酋泅趋区蛆曲躯屈驱渠取娶龋趣去圈颧权醛泉全痊拳犬券劝缺炔瘸却鹊榷确雀裙群然燃冉染瓤壤攘嚷让饶扰绕惹热壬仁人忍韧任认刃妊纫扔仍日戎茸蓉荣融熔溶容绒冗揉柔肉茹蠕儒孺如辱乳汝入褥软阮蕊瑞锐闰润若弱撒洒萨腮鳃塞赛三叁伞散桑嗓丧搔骚扫嫂瑟色涩森僧莎砂杀刹沙纱傻啥煞筛晒珊苫杉山删煽衫闪陕擅赡膳善汕扇缮墒伤商赏晌上尚裳梢捎稍烧芍勺韶少哨邵绍奢赊蛇舌舍赦摄射慑涉社设砷申呻伸身深娠绅神沈审婶甚肾慎渗声生甥牲升绳省盛剩胜圣师失狮施湿诗尸虱十石拾时什食蚀实识史矢使屎驶始式示士世柿事拭誓逝势是嗜噬适仕侍释饰氏市恃室视试收手首守寿授售受瘦兽蔬枢梳殊抒输叔舒淑疏书赎孰熟薯暑曙署蜀黍鼠属术述树束戍竖墅庶数漱恕刷耍摔衰甩帅栓拴霜双爽谁水睡税吮瞬顺舜说硕朔烁斯撕嘶思私司丝死肆寺嗣四伺似饲巳松耸怂颂送宋讼诵搜艘擞嗽苏酥俗素速粟僳塑溯宿诉肃酸蒜算虽隋随绥髓碎岁穗遂隧祟孙损笋蓑梭唆缩琐索锁所塌他它她塔獭挞蹋踏胎苔抬台泰酞太态汰坍摊贪瘫滩坛檀痰潭谭谈坦毯袒碳探叹炭汤塘搪堂棠膛唐糖倘躺淌趟烫掏涛滔绦萄桃逃淘陶讨套特藤腾疼誊梯剔踢锑提题蹄啼体替嚏惕涕剃屉天添填田甜恬舔腆挑条迢眺跳贴铁帖厅听烃汀廷停亭庭挺艇通桐酮瞳同铜彤童桶捅筒统痛偷投头透凸秃突图徒途涂屠土吐兔湍团推颓腿蜕褪退吞屯臀拖托脱鸵陀驮驼椭妥拓唾挖哇蛙洼娃瓦袜歪外豌弯湾玩顽丸烷完碗挽晚皖惋宛婉万腕汪王亡枉网往旺望忘妄威巍微危韦违桅围唯惟为潍维苇萎委伟伪尾纬未蔚味畏胃喂魏位渭谓尉慰卫瘟温蚊文闻纹吻稳紊问嗡翁瓮挝蜗涡窝我斡卧握沃巫呜钨乌污诬屋无芜梧吾吴毋武五捂午舞伍侮坞戊雾晤物勿务悟误昔熙析西硒矽晰嘻吸锡牺稀息希悉膝夕惜熄烯溪汐犀檄袭席习媳喜铣洗系隙戏细瞎虾匣霞辖暇峡侠狭下厦夏吓掀锨先仙鲜纤咸贤衔舷闲涎弦嫌显险现献县腺馅羡宪陷限线相厢镶香箱襄湘乡翔祥详想响享项巷橡像向象萧硝霄削哮嚣销消宵淆晓小孝校肖啸笑效楔些歇蝎鞋协挟携邪斜胁谐写械卸蟹懈泄泻谢屑薪芯锌欣辛新忻心信衅星腥猩惺兴刑型形邢行醒幸杏性姓兄凶胸匈汹雄熊休修羞朽嗅锈秀袖绣墟戌需虚嘘须徐许蓄酗叙旭序畜恤絮婿绪续轩喧宣悬旋玄选癣眩绚靴薛学穴雪血勋熏循旬询寻驯巡殉汛训讯逊迅压押鸦鸭呀丫芽牙蚜崖衙涯雅哑亚讶焉咽阉烟淹盐严研蜒岩延言颜阎炎沿奄掩眼衍演艳堰燕厌砚雁唁彦焰宴谚验殃央鸯秧杨扬佯疡羊洋阳氧仰痒养样漾邀腰妖瑶摇尧遥窑谣姚咬舀药要耀椰噎耶爷野冶也页掖业叶曳腋夜液一壹医揖铱依伊衣颐夷遗移仪胰疑沂宜姨彝椅蚁倚已乙矣以艺抑易邑屹亿役臆逸肄疫亦裔意毅忆义益溢诣议谊译异翼翌绎茵荫因殷音阴姻吟银淫寅饮尹引隐印英樱婴鹰应缨莹萤营荧蝇迎赢盈影颖硬映哟拥佣臃痈庸雍踊蛹咏泳涌永恿勇用幽优悠忧尤由邮铀犹油游酉有友右佑釉诱又幼迂淤于盂榆虞愚舆余俞逾鱼愉渝渔隅予娱雨与屿禹宇语羽玉域芋郁吁遇喻峪御愈欲狱育誉浴寓裕预豫驭鸳渊冤元垣袁原援辕园员圆猿源缘远苑愿怨院曰约越跃钥岳粤月悦阅耘云郧匀陨允运蕴酝晕韵孕匝砸杂栽哉灾宰载再在咱攒暂赞赃脏葬遭糟凿藻枣早澡蚤躁噪造皂灶燥责择则泽贼怎增憎曾赠扎喳渣札轧铡闸眨栅榨咋乍炸诈摘斋宅窄债寨瞻毡詹粘沾盏斩辗崭展蘸栈占战站湛绽樟章彰漳张掌涨杖丈帐账仗胀瘴障招昭找沼赵照罩兆肇召遮折哲蛰辙者锗蔗这浙珍斟真甄砧臻贞针侦枕疹诊震振镇阵蒸挣睁征狰争怔整拯正政帧症郑证芝枝支吱蜘知肢脂汁之织职直植殖执值侄址指止趾只旨纸志挚掷至致置帜峙制智秩稚质炙痔滞治窒中盅忠钟衷终种肿重仲众舟周州洲诌粥轴肘帚咒皱宙昼骤珠株蛛朱猪诸诛逐竹烛煮拄瞩嘱主著柱助蛀贮铸筑住注祝驻抓爪拽专砖转撰赚篆桩庄装妆撞壮状椎锥追赘坠缀谆准捉拙卓桌琢茁酌啄着灼浊兹咨资姿滋淄孜紫仔籽滓子自渍字鬃棕踪宗综总纵邹走奏揍租足卒族祖诅阻组钻纂嘴醉最罪尊遵昨左佐柞做作坐座"
    self.transform_dict = {}
    for hxw, jian in zip(HXW, JIAN):
      self.transform_dict[jian] = hxw

  def __call__(self, tokens, idx):
    target_token = tokens[idx]
    new_token = ''.join([self._transform(char) for char in target_token])
    if new_token == target_token:
      return None
    new_tokens = tokens[:idx] + [new_token] + tokens[idx + 1:]
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def _transform(self, char):
    if char in self.transform_dict:
      return self.transform_dict[char]
    else:
      return char


class RadicalTransform(Transform):
  def __init__(self, radical_path, max_radicals_lengths=3):
    """
    偏旁部首拆分
    """
    super().__init__()
    self.radical_map = {}
    with open(radical_path, encoding='utf-8') as f:
      for line in f:
        data = line.strip().split('\t')
        word = data[0]
        if word == '□':
          continue
        radicals = data[-1].split()
        if len(radicals) > max_radicals_lengths:
          continue
        self.radical_map[word] = radicals

  def __call__(self, tokens, idx):
    new_chars = []
    target_token = tokens[idx]
    for char in tokens[idx]:
      if char not in self.radical_map:
        new_chars.append(char)
      else:
        new_chars.extend(self.radical_map[char])
    new_token = ''.join(new_chars)
    if new_token == target_token:
      return None

    new_tokens = tokens[:idx] + [new_token] + tokens[idx + 1:]
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens


class ShapeTransform(Transform):
  def __init__(self):
    """
    形状变换，比如1lI互换, 0O、a@、eco，Ss$等
    """
    super().__init__()
    equal_sets = [
      ['1', 'l', 'i'],
      ['0', 'o', 'O'],  
      ['s', '$', ],
      ['a', '@'],  
      ['2', 'z'],
    ]
    self.transform_dict = defaultdict(list)
    for equal_set in equal_sets:
      for i in range(len(equal_set)):
        self.transform_dict[equal_set[i]].extend(equal_set[:i] + equal_set[i + 1:])  # 不算自己的方法
        # self.transform_dict[equal_set[i]].extend(equal_set)  #  带自己
    for key in self.transform_dict:
      self.transform_dict[key] = list(set(self.transform_dict[key]))

  def __call__(self, tokens, idx):
    target_token = tokens[idx]
    new_token = ''.join([self._transform(char) for char in target_token])
    if new_token == target_token:
      return None
    new_tokens = tokens[:idx] + [new_token] + tokens[idx + 1:]
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def _transform(self, char):
    if char in self.transform_dict:
      return random.choice(self.transform_dict[char])
    else:
      return char


class PronunciationTransform(Transform):
  def __init__(self, chinese_chars_file, N=5):
    """
    发音变换，换成和给定char同音的其他词
    """
    super().__init__()
    self.lazy_pinyin_dict = defaultdict(list)
    self.pinyin_dict = defaultdict(list)
    with open(chinese_chars_file, encoding='utf-8') as f:
      processed_char_set = set()
      for char in f:
        char = char.strip()
        if char in processed_char_set:
          continue
        processed_char_set.add(char)
        py = pinyin(char)
        if len(py) > 0:
          self.pinyin_dict[py[0][-1]].append(char)
        lazy_py = lazy_pinyin(char)
        if len(lazy_py) > 0:
          self.lazy_pinyin_dict[lazy_py[-1]].append(char)
    self.N = N

  def __call__(self, tokens, idx):
    target_token = tokens[idx]
    new_token = ''.join([self._transform(char) for char in target_token])
    if target_token == new_token:
      return None

    new_tokens = tokens[:idx] + [new_token] + tokens[idx + 1:]
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def _transform(self, char):
    candidates = []
    candidates.extend(self.pinyin_dict_transform(char, self.N))
    candidates.extend(self.lazy_pinyin_dict_transform(char, self.N))
    candidates = list((set(candidates) | {char}) - {char})
    if len(candidates) == 0:
      return char

    # probs = prob_calculator.calc_probs(char, candidates)
    probs = None
    new_char = np.random.choice(candidates, 1, p=probs)[0]
    return new_char

  def pinyin_dict_transform(self, char, N):
    py = pinyin(char)
    if len(py) > 0 and py[0][-1] in self.pinyin_dict:
      return self.pinyin_dict[py[0][-1]][:N]
    else:
      return []

  def lazy_pinyin_dict_transform(self, char, N):
    lazy_py = lazy_pinyin(char)
    if len(lazy_py) > 0 and lazy_py[-1] in self.lazy_pinyin_dict:
      return self.lazy_pinyin_dict[lazy_py[-1]][:N]
    else:
      return []


class StrictSameRadicalTransform(Transform):
  def __init__(self, radical_path, max_radicals_lengths=2):
    """
    构建一个字的最右子结构->字的映射表(比如妈->犸)，从而可以根据给定的字查询出有相同的右结构的字.
    Strict指字的拼音首字母还需要一致
    :param radical_path:
    """
    super().__init__()
    self.word2radical = {}
    self.radical2word = {}
    with open(radical_path, encoding='utf-8') as f:
      for line in f:
        data = line.strip().split('\t')
        word = data[0]
        if word == '□':
          continue
        py = lazy_pinyin(word)
        if not py:
          continue
        py_first_letter = py[-1][0]
        radicals = data[-1].split()
        if len(radicals) > max_radicals_lengths:
          continue

        last_radical = radicals[-1]
        if last_radical not in self.radical2word:
          self.radical2word[last_radical] = defaultdict(list)

        self.radical2word[last_radical][py_first_letter].append(word)
        self.word2radical[word] = radicals

  def __call__(self, tokens, idx):
    target_token = tokens[idx]
    new_token = ''.join([self._transform(char) for char in target_token])
    if new_token == target_token:
      return None

    new_tokens = tokens[:idx] + [new_token] + tokens[idx + 1:]
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def _transform(self, char):
    if char not in self.word2radical:
      return char
    else:
      last_radical = self.word2radical[char][-1]
      py = lazy_pinyin(char)
      if len(py) > 0 and last_radical in self.radical2word:
        py_first_letter = py[-1][0]
        return random.choice(self.radical2word[last_radical][py_first_letter])
      else:
        return char


class SimpleSameRadicalTransform(Transform):
  def __init__(self, radical_path, max_radicals_lengths=2):
    """
    构建一个字的最右子结构->字的映射表(比如妈->犸)，从而可以根据给定的字查询出有相同的右结构的字
    :param radical_path:
    """
    super().__init__()
    self.word2radical = {}
    self.radical2word = defaultdict(list)
    with open(radical_path, encoding='utf-8') as f:
      for line in f:
        data = line.strip().split('\t')
        word = data[0]
        if word == '□':
          continue
        radicals = data[-1].split()
        if len(radicals) > max_radicals_lengths:
          continue

        last_radical = radicals[-1]
        self.radical2word[last_radical].append(word)
        self.word2radical[word] = radicals

  def __call__(self, tokens, idx):
    target_token = tokens[idx]
    new_token = ''.join([self._transform(char) for char in target_token])
    if new_token == target_token:
      return None

    new_tokens = tokens[:idx] + [new_token] + tokens[idx + 1:]
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def _transform(self, char):
    if char not in self.word2radical:
      return char
    else:
      last_radical = self.word2radical[char][-1]
      if last_radical in self.radical2word:
        return random.choice(self.radical2word[last_radical])
      else:
        return char

class SequentialModel(Transform):
  def __init__(self, transforms):
    super().__init__()
    self.transforms = transforms

  def __call__(self, tokens, idx):
    new_tokens = tokens[:]
    for transform in self.transforms:
      new_tokens_ = transform(new_tokens, idx)
      if new_tokens_ is not None:
        new_tokens = new_tokens_
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def multi_ptr_trans(self, tokens, indices):
    new_tokens = tokens[:]
    for transform in self.transforms:
      new_tokens_ = transform.multi_ptr_trans(new_tokens, indices)
      if new_tokens_ is not None:
        new_tokens = new_tokens_
    if self.debug:
      self.transformed_tokens.append(new_tokens)
    return new_tokens

  def __str__(self) -> str:
    return '\t'.join([str(transform) for transform in self.transforms])
