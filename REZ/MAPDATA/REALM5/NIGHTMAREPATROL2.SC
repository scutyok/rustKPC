Name "nightmarepatrol2"

AbortOnDamage			FALSE		
AbortOnEnemySighted		FALSE		
SightRange				1600.0		
TimeOut					999.0
WalkSpeed				300
YawRate					180
CheckForEnemyInterval	0.5			
TargetPlayer						

Loop1:
OnEnemyOutOfRange_Goto	DISABLE		
OnTimerExpired_Goto		DISABLE	
OnEnemySighted_Goto		Attack				
OnDamage_Goto			FarAttack			
TimeOut		10.0

StartAnim	"Run"
TurnTo		path2x
MoveTo		path2x
TurnToAngle	180
StartAnim	"AmbA"
Wait		1.0

Goto	Loop1

FarAttack:
SightRange	2048.0

Attack:
OnDamage_Goto			DISABLE	
OnEnemySighted_Goto		DISABLE	
OnEnemyOutOfRange_Goto	Loop1

YawRate		540
TimeOut		2.5

if_RandomLessThan25_Goto	Attack_path2
if_RandomLessThan25_Goto	Attack_path1
if_RandomLessThan25_Goto	Attack_path3

Attack_path2:
TurnTo		path2x
Run
StartAnim	"Run"
OnTimerExpired_Goto		Attack_path2a
StartRandomTimer	0.5		2.0

MoveTo		path2x

Attack_path2a:
TurnToTarget	0.33	
RangedAttackTarget

if_RandomLessThan50_Goto	Attack_path3

Attack_path1:
TurnTo		path1x
Run
StartAnim	"Run"

OnTimerExpired_Goto		Attack_path1a
StartRandomTimer	0.5		2.0

MoveTo		path1x

Attack_path1a:
TurnToTarget	0.33	
RangedAttackTarget

if_RandomLessThan50_Goto	Attack_path2


Attack_path3:
TurnTo		path3x
Run
StartAnim	"Run"

OnTimerExpired_Goto		Attack_path3a
StartRandomTimer	0.5		2.0

MoveTo		path3x

Attack_path3a:
TurnToTarget	0.33	
RangedAttackTarget

if_RandomLessThan50_Goto	Attack_path1

Goto	Attack_path2

